//! Paper-stack editor with polygon-based layers. Each layer is a 2D
//! region (a `geo::MultiPolygon<f64>`) — initially the full paper
//! rectangle. Carving subtracts shapes (disks for the brush, glyph
//! contours for letters) from the top N layers, with each successive
//! layer eroded slightly inward to produce a stepped bevel.
//!
//! The mesh generator emits, for each layer, the visible "shelf"
//! (this layer's region minus the layer above) plus vertical walls
//! along the layer's interior holes. Stacked together the steps
//! read as a beveled carve.
//!
//! Press Tab to toggle the debug panel. Controls:
//!   • Left mouse drag        → apply current tool (brush)
//!   • Two-finger trackpad    → pan
//!   • Pinch                  → zoom
//!   • [ / ]                  → camera tilt
//!   • T                      → snap tilt
//!   • 1 / 2                  → select Dig / Extrude
//!   • R                      → reset camera, lights, stack
//!
//! Camera, lights, brush, palette, and tool settings persist to
//! `~/.book_ui_state.json` (saved every ~2 s).

use bevy::asset::RenderAssetUsages;
use bevy::input::ButtonInput;
use bevy::input::gestures::{PanGesture, PinchGesture};
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::light::{CascadeShadowConfigBuilder, GlobalAmbientLight};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use geo::{BooleanOps, Contains, Coord, LineString, MultiPolygon, Polygon};
use lyon::math::point as lyon_point;
use lyon::path::iterator::PathIterator;
use lyon::path::{Path as LyonPath, PathEvent};
use lyon::tessellation::{
    BuffersBuilder, FillOptions, FillRule, FillTessellator, FillVertex, VertexBuffers,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use ttf_parser::{Face, GlyphId, OutlineBuilder};

// World-unit dimensions of the paper.
const PAPER_W: f32 = 10.24;
const PAPER_H: f32 = 10.24;
const N_LAYERS: u16 = 32;
const DEFAULT_PAPER_THICKNESS: f32 = 0.018;
/// Per-layer inward inset of the carve outline. Stacking N
/// successively-eroded carves at this inset gives a chiseled bevel
/// whose slope is `pyramid_inset / paper_thickness`.
const DEFAULT_PYRAMID_INSET: f32 = 0.012;
const DEFAULT_MITER_LIMIT: f32 = 4.0;

const FONT_FALLBACKS: &[&str] = &[
    "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
];

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

// ─── Persistence ──────────────────────────────────────────────────

/// Bumped whenever the on-disk schema or default palette changes.
/// Older states get their palette discarded and recomputed from the
/// current `default_palette`, but other settings load through.
const CURRENT_STATE_VERSION: u32 = 2;

/// Snapshot of the user-facing settings (camera, lights, brush, etc.)
/// that survive across runs. The carved geometry itself is *not*
/// persisted — re-carve from the panel after restart.
#[derive(Serialize, Deserialize, Default, Debug, Clone)]
struct PersistedState {
    #[serde(default)]
    version: u32,
    camera_pan: [f32; 2],
    camera_zoom: f32,
    camera_tilt: f32,
    lights: Vec<PersistedLight>,
    ambient: f32,
    paper_thickness: f32,
    pyramid_inset: f32,
    miter_limit: f32,
    brush_radius: f32,
    brush_strength: u16,
    brush_layer_shrink: f32,
    palette_lin: Vec<[f32; 3]>,
    letters_text: String,
    letters_em: f32,
    letters_depth: u16,
    letters_bevel: bool,
    doc_max_width: f32,
    doc_line_factor: f32,
    doc_depth: u16,
    doc_bevel: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct PersistedLight {
    dir: [f32; 3],
    intensity: f32,
    color_idx: usize,
}

fn state_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".book_ui_state.json")
}

fn load_persisted_state() -> Option<PersistedState> {
    let path = state_path();
    let bytes = std::fs::read(&path).ok()?;
    match serde_json::from_slice::<PersistedState>(&bytes) {
        Ok(s) => Some(s),
        Err(e) => {
            eprintln!("[persist] failed to parse {}: {e}", path.display());
            None
        }
    }
}

fn save_persisted_state(state: &PersistedState) {
    let path = state_path();
    match serde_json::to_vec_pretty(state) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&path, json) {
                eprintln!("[persist] failed to write {}: {e}", path.display());
            }
        }
        Err(e) => eprintln!("[persist] failed to serialize: {e}"),
    }
}

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
    paper_size: Vec2,
    /// `layers[0]` = bottom layer, `layers[len-1]` = top layer.
    /// Each layer is a 2D region; carving subtracts from it.
    layers: Vec<MultiPolygon<f64>>,
    palette: Vec<Color>,
    paper_thickness: f32,
    pyramid_inset: f32,
    miter_limit: f32,
}

impl PaperStack {
    fn new(size: Vec2, n_layers: u16, palette: Vec<Color>) -> Self {
        let full = full_rect(size);
        Self {
            paper_size: size,
            layers: vec![full; n_layers as usize],
            palette,
            paper_thickness: DEFAULT_PAPER_THICKNESS,
            pyramid_inset: DEFAULT_PYRAMID_INSET,
            miter_limit: DEFAULT_MITER_LIMIT,
        }
    }
    fn n_layers(&self) -> u16 {
        self.layers.len() as u16
    }
    fn reset_full(&mut self) {
        let full = full_rect(self.paper_size);
        for layer in self.layers.iter_mut() {
            *layer = full.clone();
        }
    }
    /// World-z of the top of layer `i` (0 = bottom layer).
    fn layer_top_z(&self, i: u16) -> f32 {
        (i as f32 + 1.0) * self.paper_thickness
    }
}

/// A full-rectangle paper region in f64 coords, centered at origin.
fn full_rect(size: Vec2) -> MultiPolygon<f64> {
    let hw = size.x as f64 * 0.5;
    let hh = size.y as f64 * 0.5;
    let exterior = LineString::from(vec![
        (-hw, -hh),
        ( hw, -hh),
        ( hw,  hh),
        (-hw,  hh),
        (-hw, -hh),
    ]);
    MultiPolygon(vec![Polygon::new(exterior, vec![])])
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tool {
    Dig,
    Extrude,
}

#[derive(Resource)]
struct Brush {
    tool: Tool,
    /// Brush radius in world units (top of carve).
    radius: f32,
    /// How many layers the brush dives through at the center.
    strength: u16,
    /// Per-layer multiplicative shrink of the radius. radius_k = radius * shrink^k.
    layer_shrink: f32,
}

#[derive(Resource, Default)]
struct HoverPos(Option<Vec2>);

#[derive(Resource, Default)]
struct StrokeLast(Option<Vec2>);

#[derive(Resource, Default)]
struct StackDirty(bool);

#[derive(Component)]
struct StackMesh;

#[derive(Resource)]
struct LoadedFont(Vec<u8>);

struct NamedFont {
    name: &'static str,
    data: Vec<u8>,
}

#[derive(Resource)]
struct FontLibrary {
    fonts: Vec<NamedFont>,
}

#[derive(Clone)]
struct StyledRun {
    text: String,
    font_idx: usize,
    em: f32,
}

#[derive(Clone)]
struct DocLine {
    runs: Vec<StyledRun>,
}

#[derive(Resource)]
struct DocumentUi {
    /// World units. Roughly the on-page width of the lorem.
    max_width: f32,
    /// Multiple of the line's largest em used as line advance.
    line_factor: f32,
    depth: u16,
    bevel: bool,
}

#[derive(Resource)]
struct FormUi {
    /// Carve depth in layers for the form-field shapes.
    depth: u16,
    /// Bevel by per-layer erosion of every shape.
    bevel: bool,
}

#[derive(Resource)]
struct LettersUi {
    text: String,
    /// Em height in world units.
    em_world: f32,
    /// Carve depth in layers.
    depth: u16,
    /// If true, erode glyph by `pyramid_inset` per layer to make a
    /// chiseled bevel; otherwise straight cylindrical carve.
    bevel: bool,
}

// ─── Defaults ─────────────────────────────────────────────────────

fn default_lights() -> [LightDef; 3] {
    [
        LightDef { dir: Vec3::new(-0.55, -0.55, 0.55), intensity: 1.0, color_idx: 0 },
        LightDef { dir: Vec3::new( 0.55, -0.40, 0.55), intensity: 0.0, color_idx: 2 },
        LightDef { dir: Vec3::new( 0.0,   0.55, 0.55), intensity: 0.0, color_idx: 1 },
    ]
}

/// Layer palette. `palette[0]` is the bottom sheet, `palette[N-1]`
/// is the top sheet — that top color is what's visible on uncarved
/// paper from a top-down view, so it should look paper-like. Sweep:
///
///   bottom: deep indigo  → magenta → rose → peach → top: warm cream
///
/// Hue interpolation goes the long way (258° → 38°, +140° wrap)
/// through the magenta/red/orange side, never through cyan or green.
/// Saturation peaks in the middle layers to make the strata pop when
/// a carve exposes them; lightness rises monotonically toward the
/// top sheet.
fn default_palette(n: u16) -> Vec<Color> {
    (0..n)
        .map(|i| {
            // t=0 at the bottom layer, t=1 at the top.
            let t = i as f32 / (n.max(2) - 1) as f32;
            let h = (258.0 + 140.0 * t).rem_euclid(360.0);
            let s = 0.22 + 0.40 * (4.0 * t * (1.0 - t)).clamp(0.0, 1.0);
            let l = 0.30 + 0.58 * t;
            Color::hsl(h, s, l)
        })
        .collect()
}

// ─── App ──────────────────────────────────────────────────────────

fn main() {
    let font_data = load_font();
    let font_library = load_font_library();
    let persisted = load_persisted_state();

    let camera_ctrl = CameraControl {
        pan: persisted
            .as_ref()
            .map(|s| Vec2::from_array(s.camera_pan))
            .unwrap_or(Vec2::ZERO),
        zoom: persisted.as_ref().map(|s| s.camera_zoom).unwrap_or(1.0),
        tilt: persisted.as_ref().map(|s| s.camera_tilt).unwrap_or(0.0),
    };

    let light_ctrl = LightControl {
        lights: persisted
            .as_ref()
            .filter(|s| s.lights.len() == 3)
            .map(|s| {
                let mut arr = default_lights();
                for (i, pl) in s.lights.iter().take(3).enumerate() {
                    arr[i] = LightDef {
                        dir: Vec3::from_array(pl.dir),
                        intensity: pl.intensity,
                        color_idx: pl.color_idx,
                    };
                }
                arr
            })
            .unwrap_or_else(default_lights),
        ambient: persisted.as_ref().map(|s| s.ambient).unwrap_or(0.32),
    };

    let mut stack = PaperStack::new(
        Vec2::new(PAPER_W, PAPER_H),
        N_LAYERS,
        default_palette(N_LAYERS),
    );
    if let Some(s) = persisted.as_ref() {
        if s.paper_thickness > 0.0 {
            stack.paper_thickness = s.paper_thickness;
        }
        if s.pyramid_inset >= 0.0 {
            stack.pyramid_inset = s.pyramid_inset;
        }
        if s.miter_limit >= 1.0 {
            stack.miter_limit = s.miter_limit;
        }
        // Only restore the saved palette if the schema version matches
        // — old saves get the new default to avoid dragging stale color
        // schemes forward across redesigns.
        if s.version == CURRENT_STATE_VERSION
            && s.palette_lin.len() == stack.palette.len()
        {
            stack.palette = s
                .palette_lin
                .iter()
                .map(|c| Color::linear_rgb(c[0], c[1], c[2]))
                .collect();
        }
    }

    let brush = persisted
        .as_ref()
        .map(|s| Brush {
            tool: Tool::Dig,
            radius: s.brush_radius.max(0.05),
            strength: s.brush_strength.max(1),
            layer_shrink: s.brush_layer_shrink.clamp(0.5, 0.99),
        })
        .unwrap_or(Brush {
            tool: Tool::Dig,
            radius: 0.4,
            strength: 4,
            layer_shrink: 0.9,
        });

    let letters = persisted
        .as_ref()
        .map(|s| LettersUi {
            text: if s.letters_text.is_empty() { "Hello".into() } else { s.letters_text.clone() },
            em_world: if s.letters_em > 0.0 { s.letters_em } else { 2.0 },
            depth: s.letters_depth.max(1),
            bevel: s.letters_bevel,
        })
        .unwrap_or(LettersUi {
            text: "Hello".into(),
            em_world: 2.0,
            depth: 8,
            bevel: true,
        });

    let doc = persisted
        .as_ref()
        .map(|s| DocumentUi {
            max_width: if s.doc_max_width > 0.0 { s.doc_max_width } else { 9.0 },
            line_factor: if s.doc_line_factor >= 1.0 { s.doc_line_factor } else { 1.4 },
            depth: s.doc_depth.max(1),
            bevel: s.doc_bevel,
        })
        .unwrap_or(DocumentUi {
            max_width: 9.0,
            line_factor: 1.4,
            depth: 6,
            bevel: true,
        });

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
        .insert_resource(light_ctrl)
        .insert_resource(camera_ctrl)
        .insert_resource(GlobalAmbientLight {
            color: Color::WHITE,
            brightness: 260.0,
            affects_lightmapped_meshes: true,
        })
        .insert_resource(stack)
        .insert_resource(brush)
        .insert_resource(HoverPos::default())
        .insert_resource(StrokeLast::default())
        .insert_resource(StackDirty(true))
        .insert_resource(LoadedFont(font_data))
        .insert_resource(font_library)
        .insert_resource(letters)
        .insert_resource(doc)
        .insert_resource(FormUi { depth: 6, bevel: true })
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
                save_state_periodic,
            ),
        )
        .add_systems(EguiPrimaryContextPass, debug_panel)
        .run();
}

fn save_state_periodic(
    time: Res<Time>,
    mut last_save: Local<f64>,
    camera: Res<CameraControl>,
    light: Res<LightControl>,
    brush: Res<Brush>,
    stack: Res<PaperStack>,
    letters: Res<LettersUi>,
    doc: Res<DocumentUi>,
) {
    let now = time.elapsed_secs_f64();
    if now - *last_save < 2.0 {
        return;
    }
    *last_save = now;

    let state = PersistedState {
        version: CURRENT_STATE_VERSION,
        camera_pan: camera.pan.to_array(),
        camera_zoom: camera.zoom,
        camera_tilt: camera.tilt,
        lights: light
            .lights
            .iter()
            .map(|l| PersistedLight {
                dir: l.dir.to_array(),
                intensity: l.intensity,
                color_idx: l.color_idx,
            })
            .collect(),
        ambient: light.ambient,
        paper_thickness: stack.paper_thickness,
        pyramid_inset: stack.pyramid_inset,
        miter_limit: stack.miter_limit,
        brush_radius: brush.radius,
        brush_strength: brush.strength,
        brush_layer_shrink: brush.layer_shrink,
        palette_lin: stack
            .palette
            .iter()
            .map(|c| {
                let lc = c.to_linear();
                [lc.red, lc.green, lc.blue]
            })
            .collect(),
        letters_text: letters.text.clone(),
        letters_em: letters.em_world,
        letters_depth: letters.depth,
        letters_bevel: letters.bevel,
        doc_max_width: doc.max_width,
        doc_line_factor: doc.line_factor,
        doc_depth: doc.depth,
        doc_bevel: doc.bevel,
    };
    save_persisted_state(&state);
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
        "no font found. Set LETTERS_FONT to a TTF, or install one of: {:?}",
        FONT_FALLBACKS
    );
}

/// Collection of named macOS system fonts used for the lorem-ipsum
/// document. Each entry gives the display name and a list of paths
/// to try in order. Missing fonts are silently skipped.
fn load_font_library() -> FontLibrary {
    const CANDIDATES: &[(&str, &[&str])] = &[
        ("Georgia",        &["/System/Library/Fonts/Supplemental/Georgia.ttf"]),
        ("Georgia Bold",   &["/System/Library/Fonts/Supplemental/Georgia Bold.ttf"]),
        ("Georgia Italic", &["/System/Library/Fonts/Supplemental/Georgia Italic.ttf"]),
        ("Times",          &["/System/Library/Fonts/Supplemental/Times New Roman.ttf"]),
        ("Times Bold",     &["/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"]),
        ("Times Italic",   &["/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf"]),
        ("Arial",          &["/System/Library/Fonts/Supplemental/Arial.ttf",
                             "/Library/Fonts/Arial.ttf"]),
        ("Arial Bold",     &["/System/Library/Fonts/Supplemental/Arial Bold.ttf"]),
        ("Comic Sans",     &["/System/Library/Fonts/Supplemental/Comic Sans MS.ttf"]),
        ("Comic Sans Bold",&["/System/Library/Fonts/Supplemental/Comic Sans MS Bold.ttf"]),
        ("Courier",        &["/System/Library/Fonts/Supplemental/Courier New.ttf"]),
        ("Courier Bold",   &["/System/Library/Fonts/Supplemental/Courier New Bold.ttf"]),
        ("Verdana",        &["/System/Library/Fonts/Supplemental/Verdana.ttf"]),
        ("Verdana Italic", &["/System/Library/Fonts/Supplemental/Verdana Italic.ttf"]),
        ("Trebuchet",      &["/System/Library/Fonts/Supplemental/Trebuchet MS.ttf"]),
        ("Impact",         &["/System/Library/Fonts/Supplemental/Impact.ttf"]),
    ];
    let mut fonts = Vec::new();
    for (name, paths) in CANDIDATES {
        for p in *paths {
            if let Ok(data) = std::fs::read(p) {
                fonts.push(NamedFont { name, data });
                break;
            }
        }
    }
    if fonts.is_empty() {
        // Fall back to whatever single font we found, so the lorem
        // tool still has something to work with.
        if let Some(data) = FONT_FALLBACKS
            .iter()
            .find_map(|p| std::fs::read(p).ok())
        {
            fonts.push(NamedFont { name: "fallback", data });
        }
    }
    FontLibrary { fonts }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    stack: Res<PaperStack>,
) {
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

    let stack_mat = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        perceptual_roughness: 0.92,
        metallic: 0.0,
        reflectance: 0.04,
        double_sided: true,
        cull_mode: None,
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
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut colors: Vec<[f32; 4]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let pt = stack.paper_thickness;
    let n = stack.layers.len();
    let empty_mp: MultiPolygon<f64> = MultiPolygon(vec![]);

    for i in 0..n {
        let layer = &stack.layers[i];
        let above: &MultiPolygon<f64> = if i + 1 < n { &stack.layers[i + 1] } else { &empty_mp };

        // Visible top of this layer = layer minus what's above it.
        let exposed = layer.difference(above);
        let color = color_to_rgba(stack.palette[i % stack.palette.len()]);
        let tris_before = indices.len();
        if !exposed.0.is_empty() {
            let z = (i as f32 + 1.0) * pt;
            tessellate_region(
                &exposed,
                z,
                color,
                &mut positions,
                &mut normals,
                &mut colors,
                &mut indices,
            );
        }
        let _ = tris_before;

        // Vertical walls along this layer's interior holes.
        let z_top = (i as f32 + 1.0) * pt;
        let z_bot = i as f32 * pt;
        for poly in &layer.0 {
            for hole in poly.interiors() {
                emit_wall_for_ring(
                    hole,
                    /* is_interior_hole */ true,
                    z_top,
                    z_bot,
                    color,
                    &mut positions,
                    &mut normals,
                    &mut colors,
                    &mut indices,
                );
            }
        }
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

fn tessellate_region(
    region: &MultiPolygon<f64>,
    z: f32,
    color: [f32; 4],
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    // Tessellate each polygon (exterior + its own holes) on its own
    // path. Lyon's FillTessellator gets unhappy when one path mixes
    // many disjoint exteriors with their nested island sub-polygons,
    // so we keep each topological component independent.
    let mut tess = FillTessellator::new();
    let opts = FillOptions::default()
        .with_fill_rule(FillRule::NonZero)
        .with_tolerance(0.01);
    for poly in &region.0 {
        let mut pb = LyonPath::builder();
        if !push_ring_to_path(poly.exterior(), &mut pb) {
            continue;
        }
        for hole in poly.interiors() {
            push_ring_to_path(hole, &mut pb);
        }
        let path = pb.build();

        let mut buffers: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
        if tess
            .tessellate_path(
                &path,
                &opts,
                &mut BuffersBuilder::new(&mut buffers, |v: FillVertex| {
                    let p = v.position();
                    [p.x, p.y]
                }),
            )
            .is_err()
        {
            continue;
        }

        let base = positions.len() as u32;
        for v in &buffers.vertices {
            positions.push([v[0], v[1], z]);
            normals.push([0.0, 0.0, 1.0]);
            colors.push(color);
        }
        for idx in &buffers.indices {
            indices.push(base + idx);
        }
    }
}

/// Append a closed ring to a lyon path. Returns true if the ring had
/// at least 3 distinct points.
fn push_ring_to_path(ring: &LineString<f64>, pb: &mut lyon::path::path::Builder) -> bool {
    let coords = &ring.0;
    // Drop the closing duplicate if present.
    let n_raw = coords.len();
    if n_raw < 3 {
        return false;
    }
    let count = if (coords[0].x - coords[n_raw - 1].x).abs() < 1e-9
        && (coords[0].y - coords[n_raw - 1].y).abs() < 1e-9
    {
        n_raw - 1
    } else {
        n_raw
    };
    if count < 3 {
        return false;
    }
    pb.begin(lyon_point(coords[0].x as f32, coords[0].y as f32));
    for c in &coords[1..count] {
        pb.line_to(lyon_point(c.x as f32, c.y as f32));
    }
    pb.end(true);
    true
}

/// Emit a vertical wall along a closed ring. For interior holes, the
/// outward normal points into the void (right of CW walking direction).
fn emit_wall_for_ring(
    ring: &LineString<f64>,
    is_interior_hole: bool,
    z_top: f32,
    z_bot: f32,
    color: [f32; 4],
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    let pts = ring_points(ring);
    let n = pts.len();
    if n < 3 {
        return;
    }

    for i in 0..n {
        let j = (i + 1) % n;
        let pi = pts[i];
        let pj = pts[j];

        let edge = (pj - pi).try_normalize().unwrap_or(Vec2::X);
        // For a CW interior hole, void is on the RIGHT of walking dir.
        // For a CCW exterior, void is on the LEFT (outside).
        let perp = if is_interior_hole {
            Vec2::new(edge.y, -edge.x)
        } else {
            Vec2::new(-edge.y, edge.x)
        };
        let normal = [perp.x, perp.y, 0.0];

        let p_top_i = [pi.x, pi.y, z_top];
        let p_top_j = [pj.x, pj.y, z_top];
        let p_bot_i = [pi.x, pi.y, z_bot];
        let p_bot_j = [pj.x, pj.y, z_bot];

        let base = positions.len() as u32;
        for v in [p_top_i, p_bot_i, p_bot_j, p_top_j] {
            positions.push(v);
            normals.push(normal);
            colors.push(color);
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// Returns ring points without the closing duplicate.
fn ring_points(ring: &LineString<f64>) -> Vec<Vec2> {
    let coords = &ring.0;
    let n_raw = coords.len();
    if n_raw == 0 {
        return Vec::new();
    }
    let count = if n_raw >= 2
        && (coords[0].x - coords[n_raw - 1].x).abs() < 1e-9
        && (coords[0].y - coords[n_raw - 1].y).abs() < 1e-9
    {
        n_raw - 1
    } else {
        n_raw
    };
    coords[..count]
        .iter()
        .map(|c| Vec2::new(c.x as f32, c.y as f32))
        .collect()
}

fn color_to_rgba(c: Color) -> [f32; 4] {
    let lc = c.to_linear();
    [lc.red, lc.green, lc.blue, lc.alpha]
}

// ─── Carve operations ─────────────────────────────────────────────

/// Subtract a stepped sequence of shapes from the top-down layers.
/// `shapes[k]` (if `Some`) is subtracted from layer `top-k`.
fn carve_stepped(stack: &mut PaperStack, shapes: &[Option<MultiPolygon<f64>>]) {
    if stack.layers.is_empty() {
        return;
    }
    let top = stack.layers.len() - 1;
    for (k, maybe_shape) in shapes.iter().enumerate() {
        if k > top {
            break;
        }
        if let Some(shape) = maybe_shape {
            if shape.0.is_empty() {
                continue;
            }
            let layer_idx = top - k;
            stack.layers[layer_idx] = stack.layers[layer_idx].difference(shape);
        }
    }
}

/// Add (union) shapes onto the top-down layers — refills carved-away
/// material. `shapes[k]` (if `Some`) is unioned into layer `top-k`.
fn extrude_stepped(stack: &mut PaperStack, shapes: &[Option<MultiPolygon<f64>>]) {
    if stack.layers.is_empty() {
        return;
    }
    let top = stack.layers.len() - 1;
    for (k, maybe_shape) in shapes.iter().enumerate() {
        if k > top {
            break;
        }
        if let Some(shape) = maybe_shape {
            if shape.0.is_empty() {
                continue;
            }
            let layer_idx = top - k;
            stack.layers[layer_idx] = stack.layers[layer_idx].union(shape);
        }
    }
}

fn circle_polygon(center: Vec2, radius: f32, segments: u32) -> Polygon<f64> {
    let mut pts: Vec<(f64, f64)> = Vec::with_capacity(segments as usize + 1);
    for i in 0..segments {
        let theta = i as f32 / segments as f32 * std::f32::consts::TAU;
        pts.push((
            (center.x + theta.cos() * radius) as f64,
            (center.y + theta.sin() * radius) as f64,
        ));
    }
    pts.push(pts[0]);
    Polygon::new(LineString::from(pts), vec![])
}

fn carve_brush(stack: &mut PaperStack, brush: &Brush, world_pos: Vec2) {
    let n = stack.layers.len();
    let depth = (brush.strength as usize).min(n);
    let mut shapes: Vec<Option<MultiPolygon<f64>>> = Vec::with_capacity(depth);
    let shrink = brush.layer_shrink.clamp(0.05, 0.99);
    for k in 0..depth {
        let r = brush.radius * shrink.powi(k as i32);
        if r < 0.002 {
            break;
        }
        let circ = circle_polygon(world_pos, r, 48);
        shapes.push(Some(MultiPolygon(vec![circ])));
    }
    match brush.tool {
        Tool::Dig => carve_stepped(stack, &shapes),
        Tool::Extrude => extrude_stepped(stack, &shapes),
    }
}

// ─── Glyph parsing ────────────────────────────────────────────────

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

fn ops_to_path(ops: &[Op]) -> LyonPath {
    let mut pb = LyonPath::builder();
    let mut started = false;
    for op in ops {
        match op {
            Op::MoveTo(p) => {
                if started {
                    pb.end(false);
                }
                pb.begin(lyon_point(p.x, p.y));
                started = true;
            }
            Op::LineTo(p) => {
                pb.line_to(lyon_point(p.x, p.y));
            }
            Op::QuadTo(c, p) => {
                pb.quadratic_bezier_to(lyon_point(c.x, c.y), lyon_point(p.x, p.y));
            }
            Op::CurveTo(c1, c2, p) => {
                pb.cubic_bezier_to(
                    lyon_point(c1.x, c1.y),
                    lyon_point(c2.x, c2.y),
                    lyon_point(p.x, p.y),
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

fn outline_glyph_to_path(face: &Face, glyph_id: GlyphId, scale: f32, offset: Vec2) -> LyonPath {
    let mut col = OpCollector { ops: Vec::new(), scale, offset };
    face.outline_glyph(glyph_id, &mut col);
    ops_to_path(&col.ops)
}

fn flatten_path_to_contours(path: &LyonPath, tol: f32) -> Vec<Vec<Vec2>> {
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
            PathEvent::Quadratic { .. } | PathEvent::Cubic { .. } => {}
            PathEvent::End { close, .. } => {
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

fn signed_area(pts: &[Vec2]) -> f32 {
    let mut a = 0.0;
    let n = pts.len();
    for i in 0..n {
        let j = (i + 1) % n;
        a += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
    }
    a * 0.5
}

fn point_in_ring(p: Vec2, ring: &[Vec2]) -> bool {
    let mut inside = false;
    let n = ring.len();
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (ring[i].x, ring[i].y);
        let (xj, yj) = (ring[j].x, ring[j].y);
        if (yi > p.y) != (yj > p.y) {
            let intersect_x = (xj - xi) * (p.y - yi) / (yj - yi) + xi;
            if p.x < intersect_x {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

fn linestring_from_vecs(pts: &[Vec2]) -> LineString<f64> {
    let mut coords: Vec<(f64, f64)> = pts.iter().map(|p| (p.x as f64, p.y as f64)).collect();
    if let Some(first) = coords.first().copied() {
        coords.push(first);
    }
    LineString::from(coords)
}

fn ensure_ccw(mut pts: Vec<Vec2>) -> Vec<Vec2> {
    if signed_area(&pts) < 0.0 {
        pts.reverse();
    }
    pts
}

fn ensure_cw(mut pts: Vec<Vec2>) -> Vec<Vec2> {
    if signed_area(&pts) > 0.0 {
        pts.reverse();
    }
    pts
}

/// Group a glyph's contours into geo polygons. Classifies outers vs
/// holes by **containment depth** rather than winding direction, so
/// it works regardless of whether the font (or ttf-parser) delivers
/// outers as CCW or CW. Output rings are re-oriented to geo's
/// canonical convention: exterior CCW, interior CW.
fn build_glyph_polygons(contours: &[Vec<Vec2>]) -> Vec<Polygon<f64>> {
    let valid: Vec<&Vec<Vec2>> = contours.iter().filter(|c| c.len() >= 3).collect();
    let n = valid.len();
    if n == 0 {
        return Vec::new();
    }

    // depth[i] = number of OTHER contours that contain valid[i][0].
    // depth 0 = top-level outer; depth 1 = hole; depth 2 = island; etc.
    let mut depths = vec![0usize; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if point_in_ring(valid[i][0], valid[j]) {
                depths[i] += 1;
            }
        }
    }

    let mut polygons = Vec::new();
    for i in 0..n {
        if depths[i] % 2 != 0 {
            continue; // odd depth = hole, emitted as a hole of its parent
        }
        let outer_pts = ensure_ccw(valid[i].clone());
        let mut holes: Vec<LineString<f64>> = Vec::new();
        for j in 0..n {
            if i == j || depths[j] != depths[i] + 1 {
                continue;
            }
            // Hole must be directly contained in *this* outer, not any
            // other outer at the same depth.
            if point_in_ring(valid[j][0], &outer_pts) {
                let hole_pts = ensure_cw(valid[j].clone());
                holes.push(linestring_from_vecs(&hole_pts));
            }
        }
        polygons.push(Polygon::new(linestring_from_vecs(&outer_pts), holes));
    }
    polygons
}

/// Lay out `text` and return the union of all glyph polygons,
/// centered on the origin in world space.
fn build_text_polygons(
    font_data: &[u8],
    text: &str,
    em_world: f32,
    flatten_tol: f32,
) -> MultiPolygon<f64> {
    let face = match Face::parse(font_data, 0) {
        Ok(f) => f,
        Err(_) => return MultiPolygon(vec![]),
    };
    let upem = face.units_per_em() as f32;
    let scale = em_world / upem;

    let mut all: Vec<Polygon<f64>> = Vec::new();
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
        for c in &contours {
            for p in c {
                bounds_min = bounds_min.min(*p);
                bounds_max = bounds_max.max(*p);
            }
        }
        let glyph_polys = build_glyph_polygons(&contours);
        all.extend(glyph_polys);
        x_cursor += advance * scale;
    }

    if !bounds_min.is_finite() {
        return MultiPolygon(vec![]);
    }
    let center = (bounds_min + bounds_max) * 0.5;
    let cx = center.x as f64;
    let cy = center.y as f64;

    // Center on origin and assemble. Glyphs are typically disjoint so
    // a plain MultiPolygon is fine — running a union is fragile when
    // serifs touch at a single vertex and is unnecessary otherwise.
    let recentered: Vec<Polygon<f64>> = all
        .into_iter()
        .map(|poly| translate_polygon(&poly, -cx, -cy))
        .collect();
    MultiPolygon(recentered)
}

fn translate_polygon(poly: &Polygon<f64>, dx: f64, dy: f64) -> Polygon<f64> {
    let translate_ls = |ls: &LineString<f64>| -> LineString<f64> {
        LineString::from(
            ls.0.iter()
                .map(|c| (c.x + dx, c.y + dy))
                .collect::<Vec<_>>(),
        )
    };
    Polygon::new(
        translate_ls(poly.exterior()),
        poly.interiors().iter().map(translate_ls).collect(),
    )
}

// ─── Inward miter offset (for glyph erosion per layer) ────────────

fn miter_offset_left_f64(pts: &[Coord<f64>], inset: f64, miter_limit: f64) -> Vec<Coord<f64>> {
    let n = pts.len();
    if n < 2 {
        return pts.to_vec();
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let prev = pts[(i + n - 1) % n];
        let curr = pts[i];
        let next = pts[(i + 1) % n];
        let e_in = norm2_f64(curr.x - prev.x, curr.y - prev.y);
        let e_out = norm2_f64(next.x - curr.x, next.y - curr.y);
        let n_in = (-e_in.1, e_in.0);
        let n_out = (-e_out.1, e_out.0);
        let bx = n_in.0 + n_out.0;
        let by = n_in.1 + n_out.1;
        let blen = (bx * bx + by * by).sqrt();
        let (bxn, byn) = if blen > 1e-12 {
            (bx / blen, by / blen)
        } else {
            n_in
        };
        let cos_half = (bxn * n_in.0 + byn * n_in.1).max(1.0 / miter_limit.max(1.0));
        let miter_len = inset / cos_half;
        out.push(Coord {
            x: curr.x + bxn * miter_len,
            y: curr.y + byn * miter_len,
        });
    }
    out
}

fn norm2_f64(x: f64, y: f64) -> (f64, f64) {
    let len = (x * x + y * y).sqrt();
    if len < 1e-12 {
        (1.0, 0.0)
    } else {
        (x / len, y / len)
    }
}

/// Erode (shrink) every polygon in a MultiPolygon inward by `inset`.
/// Then union the results back together so the cleanup of any
/// self-intersections produced by the offset is left to geo.
fn erode_multipolygon(mp: &MultiPolygon<f64>, inset: f64, miter_limit: f64) -> MultiPolygon<f64> {
    let mut out = MultiPolygon(vec![]);
    for poly in &mp.0 {
        let ext_pts = ring_to_coords(poly.exterior());
        if ext_pts.len() < 3 {
            continue;
        }
        let new_ext = miter_offset_left_f64(&ext_pts, inset, miter_limit);
        let new_holes: Vec<LineString<f64>> = poly
            .interiors()
            .iter()
            .filter_map(|h| {
                let pts = ring_to_coords(h);
                if pts.len() < 3 {
                    return None;
                }
                // Holes are CW: the polygon fill is on the LEFT of the
                // walking direction (same as exterior). So shrinking
                // the polygon means offsetting holes LEFT too — this
                // expands the hole inward into the fill.
                let new_pts = miter_offset_left_f64(&pts, inset, miter_limit);
                Some(coords_to_closed_linestring(&new_pts))
            })
            .collect();
        let exterior = coords_to_closed_linestring(&new_ext);
        let new_poly = Polygon::new(exterior, new_holes);
        out = out.union(&MultiPolygon(vec![new_poly]));
    }
    out
}

fn ring_to_coords(ring: &LineString<f64>) -> Vec<Coord<f64>> {
    let coords = &ring.0;
    let n = coords.len();
    if n == 0 {
        return Vec::new();
    }
    let count = if n >= 2
        && (coords[0].x - coords[n - 1].x).abs() < 1e-9
        && (coords[0].y - coords[n - 1].y).abs() < 1e-9
    {
        n - 1
    } else {
        n
    };
    coords[..count].to_vec()
}

fn coords_to_closed_linestring(pts: &[Coord<f64>]) -> LineString<f64> {
    let mut v: Vec<(f64, f64)> = pts.iter().map(|c| (c.x, c.y)).collect();
    if let Some(first) = v.first().copied() {
        v.push(first);
    }
    LineString::from(v)
}

// ─── Multi-font document layout ───────────────────────────────────

const LOREM: &str = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, \
                     sed do eiusmod tempor incididunt ut labore et dolore magna \
                     aliqua. Ut enim ad minim veniam, quis nostrud exercitation \
                     ullamco laboris nisi ut aliquip ex ea commodo consequat.";

/// Produce a multi-font, multi-size styled document. Cycles fonts
/// per word and varies em sizes to make the typographic mixture
/// obvious.
fn lorem_ipsum_doc(font_lib: &FontLibrary, words_per_line: usize) -> Vec<DocLine> {
    if font_lib.fonts.is_empty() {
        return Vec::new();
    }
    let words: Vec<&str> = LOREM.split_whitespace().collect();
    let em_cycle = [0.55_f32, 0.45, 0.62, 0.5, 0.4, 0.58, 0.5];
    let mut lines: Vec<DocLine> = Vec::new();
    let mut current = DocLine { runs: Vec::new() };
    let mut count = 0;
    for (i, word) in words.iter().enumerate() {
        let font_idx = i % font_lib.fonts.len();
        let em = em_cycle[i % em_cycle.len()];
        let prefix = if count > 0 { " " } else { "" };
        current.runs.push(StyledRun {
            text: format!("{prefix}{word}"),
            font_idx,
            em,
        });
        count += 1;
        if count >= words_per_line {
            lines.push(std::mem::replace(&mut current, DocLine { runs: Vec::new() }));
            count = 0;
        }
    }
    if !current.runs.is_empty() {
        lines.push(current);
    }
    lines
}

/// Lay a `Vec<DocLine>` out into a centered MultiPolygon. Glyphs
/// from different fonts and ems sit on a shared baseline per line;
/// each line advances by `line_factor * max_em_on_this_line`.
/// Words wrap if a line's run total exceeds `max_width`.
fn build_doc_polygons(
    font_lib: &FontLibrary,
    lines: &[DocLine],
    max_width: f32,
    line_factor: f32,
    flatten_tol: f32,
) -> MultiPolygon<f64> {
    let mut all_polys: Vec<Polygon<f64>> = Vec::new();
    let mut bounds_min = Vec2::splat(f32::INFINITY);
    let mut bounds_max = Vec2::splat(f32::NEG_INFINITY);

    let mut y_baseline = 0.0_f32;

    for doc_line in lines {
        let max_em = doc_line
            .runs
            .iter()
            .map(|r| r.em)
            .fold(0.0_f32, f32::max);
        let line_advance = max_em * line_factor;
        // Wrap state for this line.
        let mut x_cursor = 0.0_f32;
        let mut wrapped_lines = 0;

        for run in &doc_line.runs {
            let face = match Face::parse(&font_lib.fonts[run.font_idx].data, 0) {
                Ok(f) => f,
                Err(_) => continue,
            };
            let upem = face.units_per_em() as f32;
            let scale = run.em / upem;

            // Tokenize on whitespace, preserving leading space if the
            // run text starts with one (between-word separator).
            let leading_space = run.text.starts_with(char::is_whitespace);
            let words: Vec<&str> = run.text.split_whitespace().collect();
            for (wi, word) in words.iter().enumerate() {
                let word_w = measure_word(&face, word, scale);
                let space_w = match face.glyph_index(' ') {
                    Some(gid) => face.glyph_hor_advance(gid).unwrap_or(0) as f32 * scale,
                    None => run.em * 0.25,
                };

                if (wi > 0 || leading_space) && x_cursor > 0.0 {
                    x_cursor += space_w;
                }
                if x_cursor > 0.0 && x_cursor + word_w > max_width {
                    x_cursor = 0.0;
                    y_baseline -= line_advance;
                    wrapped_lines += 1;
                }

                for ch in word.chars() {
                    let Some(gid) = face.glyph_index(ch) else { continue };
                    let advance = face.glyph_hor_advance(gid).unwrap_or(0) as f32 * scale;
                    let path = outline_glyph_to_path(
                        &face,
                        gid,
                        scale,
                        Vec2::new(x_cursor, y_baseline),
                    );
                    let contours = flatten_path_to_contours(&path, flatten_tol);
                    for c in &contours {
                        for p in c {
                            bounds_min = bounds_min.min(*p);
                            bounds_max = bounds_max.max(*p);
                        }
                    }
                    let glyph_polys = build_glyph_polygons(&contours);
                    all_polys.extend(glyph_polys);
                    x_cursor += advance;
                }
            }
            let _ = wrapped_lines;
        }
        y_baseline -= line_advance;
    }

    if !bounds_min.is_finite() {
        return MultiPolygon(vec![]);
    }
    let center = (bounds_min + bounds_max) * 0.5;
    let cx = center.x as f64;
    let cy = center.y as f64;
    let recentered: Vec<Polygon<f64>> = all_polys
        .into_iter()
        .map(|poly| translate_polygon(&poly, -cx, -cy))
        .collect();
    MultiPolygon(recentered)
}

fn measure_word(face: &Face, word: &str, scale: f32) -> f32 {
    word.chars()
        .filter_map(|ch| face.glyph_index(ch))
        .map(|gid| face.glyph_hor_advance(gid).unwrap_or(0) as f32 * scale)
        .sum()
}

fn carve_document(
    stack: &mut PaperStack,
    font_lib: &FontLibrary,
    lines: &[DocLine],
    max_width: f32,
    line_factor: f32,
    depth: u16,
    bevel: bool,
) {
    let flat_tol = (stack.pyramid_inset * 0.5).max(0.005);
    let polys = build_doc_polygons(font_lib, lines, max_width, line_factor, flat_tol);
    if polys.0.is_empty() {
        return;
    }
    let n = stack.layers.len();
    let depth = (depth as usize).min(n);
    let mut shapes: Vec<Option<MultiPolygon<f64>>> = Vec::with_capacity(depth);
    let mut current = polys;
    for k in 0..depth {
        if current.0.is_empty() {
            break;
        }
        shapes.push(Some(current.clone()));
        if bevel && k + 1 < depth {
            current = erode_multipolygon(
                &current,
                stack.pyramid_inset as f64,
                stack.miter_limit as f64,
            );
        }
    }
    carve_stepped(stack, &shapes);
}

/// Carve a text string. If `bevel` is true, each successive layer's
/// shape is eroded by `pyramid_inset`, producing a chiseled bevel.
fn carve_text(stack: &mut PaperStack, font_data: &[u8], text: &str, em_world: f32, depth: u16, bevel: bool) {
    let flat_tol = (stack.pyramid_inset * 0.5).max(0.005);
    let glyphs = build_text_polygons(font_data, text, em_world, flat_tol);
    if glyphs.0.is_empty() {
        return;
    }
    let n = stack.layers.len();
    let depth = (depth as usize).min(n);
    let mut shapes: Vec<Option<MultiPolygon<f64>>> = Vec::with_capacity(depth);
    let mut current = glyphs;
    for k in 0..depth {
        if current.0.is_empty() {
            break;
        }
        shapes.push(Some(current.clone()));
        if bevel && k + 1 < depth {
            current = erode_multipolygon(
                &current,
                stack.pyramid_inset as f64,
                stack.miter_limit as f64,
            );
        }
    }
    carve_stepped(stack, &shapes);
}

// ─── Form fields (lines + UI shapes) ──────────────────────────────

fn rect_polygon(cx: f32, cy: f32, w: f32, h: f32) -> Polygon<f64> {
    let hw = w as f64 * 0.5;
    let hh = h as f64 * 0.5;
    let cx = cx as f64;
    let cy = cy as f64;
    Polygon::new(
        LineString::from(vec![
            (cx - hw, cy - hh),
            (cx + hw, cy - hh),
            (cx + hw, cy + hh),
            (cx - hw, cy + hh),
            (cx - hw, cy - hh),
        ]),
        vec![],
    )
}

fn rounded_rect_polygon(cx: f32, cy: f32, w: f32, h: f32, r: f32) -> Polygon<f64> {
    let r = r.min(w * 0.5).min(h * 0.5).max(0.0);
    if r < 1e-4 {
        return rect_polygon(cx, cy, w, h);
    }
    let segs = 8u32;
    let hw = (w * 0.5 - r) as f64;
    let hh = (h * 0.5 - r) as f64;
    let cx = cx as f64;
    let cy = cy as f64;
    let r = r as f64;

    let mut pts: Vec<(f64, f64)> = Vec::new();
    let mut corner = |center: (f64, f64), start: f64| {
        for i in 0..=segs {
            let t = i as f64 / segs as f64;
            let theta = start + t * std::f64::consts::FRAC_PI_2;
            pts.push((center.0 + theta.cos() * r, center.1 + theta.sin() * r));
        }
    };
    // CCW around the rect: bottom-right corner → top-right → top-left → bottom-left.
    corner((cx + hw, cy - hh), -std::f64::consts::FRAC_PI_2);
    corner((cx + hw, cy + hh), 0.0);
    corner((cx - hw, cy + hh), std::f64::consts::FRAC_PI_2);
    corner((cx - hw, cy - hh), std::f64::consts::PI);
    let first = pts[0];
    pts.push(first);
    Polygon::new(LineString::from(pts), vec![])
}

fn circle_polygon_at(cx: f32, cy: f32, r: f32, segments: u32) -> Polygon<f64> {
    circle_polygon(Vec2::new(cx, cy), r, segments)
}

fn triangle_polygon(p1: Vec2, p2: Vec2, p3: Vec2) -> Polygon<f64> {
    Polygon::new(
        LineString::from(vec![
            (p1.x as f64, p1.y as f64),
            (p2.x as f64, p2.y as f64),
            (p3.x as f64, p3.y as f64),
            (p1.x as f64, p1.y as f64),
        ]),
        vec![],
    )
}

/// Build a panel of form-style UI primitives carved into the paper.
/// All shapes are *filled* polygons that get subtracted from the
/// stack — so each one carves a depression in the shape of the
/// affordance.
fn build_form_polygons() -> MultiPolygon<f64> {
    let mut polys: Vec<Polygon<f64>> = Vec::new();

    // Two horizontal divider lines (top and bottom of the form area).
    polys.push(rect_polygon(0.0, 3.6, 7.0, 0.04));
    polys.push(rect_polygon(0.0, -3.6, 7.0, 0.04));

    // Two text-input fields stacked vertically — long pill-rects.
    polys.push(rounded_rect_polygon(0.0, 2.7, 6.0, 0.55, 0.12));
    polys.push(rounded_rect_polygon(0.0, 1.85, 6.0, 0.55, 0.12));

    // Three radio-button circles in a row.
    for i in 0..3 {
        let x = -2.0 + i as f32 * 2.0;
        polys.push(circle_polygon_at(x, 0.85, 0.22, 48));
    }

    // Three checkbox squares in a row (slightly rounded).
    for i in 0..3 {
        let x = -2.0 + i as f32 * 2.0;
        polys.push(rounded_rect_polygon(x, -0.1, 0.45, 0.45, 0.06));
    }

    // Dropdown — wide pill-rect with a chevron triangle in the right.
    polys.push(rounded_rect_polygon(0.0, -1.2, 6.0, 0.55, 0.12));
    let chev_x = 2.5;
    let chev_y = -1.2;
    polys.push(triangle_polygon(
        Vec2::new(chev_x - 0.18, chev_y + 0.10),
        Vec2::new(chev_x + 0.18, chev_y + 0.10),
        Vec2::new(chev_x,        chev_y - 0.18),
    ));

    // Submit-style button — small rounded rect.
    polys.push(rounded_rect_polygon(0.0, -2.4, 1.8, 0.65, 0.18));

    MultiPolygon(polys)
}

fn carve_form(stack: &mut PaperStack, depth: u16, bevel: bool) {
    let form = build_form_polygons();
    if form.0.is_empty() {
        return;
    }
    let n = stack.layers.len();
    let depth = (depth as usize).min(n);
    let mut shapes: Vec<Option<MultiPolygon<f64>>> = Vec::with_capacity(depth);
    let mut current = form;
    for k in 0..depth {
        if current.0.is_empty() {
            break;
        }
        shapes.push(Some(current.clone()));
        if bevel && k + 1 < depth {
            current = erode_multipolygon(
                &current,
                stack.pyramid_inset as f64,
                stack.miter_limit as f64,
            );
        }
    }
    carve_stepped(stack, &shapes);
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
    mut pinch: MessageReader<PinchGesture>,
    mut pan_gesture: MessageReader<PanGesture>,
    time: Res<Time>,
    mut camera: ResMut<CameraControl>,
    mut brush: ResMut<Brush>,
    mut stack: ResMut<PaperStack>,
    mut dirty: ResMut<StackDirty>,
    mut light: ResMut<LightControl>,
    mut contexts: EguiContexts,
) {
    let dt = time.delta_secs();

    // egui swallows trackpad input over its panels.
    let egui_has_pointer = contexts
        .ctx_mut()
        .map(|c| c.is_pointer_over_area() || c.wants_pointer_input())
        .unwrap_or(false);

    if keys.just_pressed(KeyCode::KeyR) {
        camera.pan = Vec2::ZERO;
        camera.zoom = 1.0;
        camera.tilt = 0.0;
        light.lights = default_lights();
        light.ambient = 0.32;
        stack.reset_full();
        dirty.0 = true;
    }

    if keys.just_pressed(KeyCode::Digit1) { brush.tool = Tool::Dig; }
    if keys.just_pressed(KeyCode::Digit2) { brush.tool = Tool::Extrude; }

    // Pan: macOS sends two-finger trackpad scroll as MouseWheel events,
    // and (on some configs) also as PanGesture events. Read both and
    // apply whichever fires. Skip when egui owns the pointer.
    let mut pan_delta = Vec2::ZERO;
    for event in wheel.read() {
        if egui_has_pointer { continue; }
        let factor = match event.unit {
            MouseScrollUnit::Pixel => 0.005,
            MouseScrollUnit::Line  => 0.10,
        };
        // Natural-scroll convention: scroll up → camera moves up
        // (content scrolls down on screen).
        pan_delta.x += event.x * factor;
        pan_delta.y += event.y * factor;
    }
    for event in pan_gesture.read() {
        if egui_has_pointer { continue; }
        pan_delta.x += event.0.x * 0.005;
        pan_delta.y += event.0.y * 0.005;
    }
    if pan_delta != Vec2::ZERO {
        let z = camera.zoom;
        camera.pan += pan_delta / z;
    }

    // Pinch: positive delta = pinch out = zoom in.
    for event in pinch.read() {
        if egui_has_pointer { continue; }
        camera.zoom = (camera.zoom * (1.0 + event.0)).clamp(0.2, 8.0);
    }

    // Tilt — keyboard fallback; rotate gesture left as a future hook.
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
    mut hover: ResMut<HoverPos>,
    mut contexts: EguiContexts,
) {
    if let Ok(ctx) = contexts.ctx_mut() {
        if ctx.is_pointer_over_area() {
            hover.0 = None;
            return;
        }
    }
    let Ok(window) = windows.single() else { hover.0 = None; return; };
    let Some(cursor) = window.cursor_position() else { hover.0 = None; return; };
    let Ok((camera, cam_tf)) = cameras.single() else { hover.0 = None; return; };
    let Ok(ray) = camera.viewport_to_world(cam_tf, cursor) else { hover.0 = None; return; };
    hover.0 = pick_top_visible(ray, &stack);
}

fn pick_top_visible(ray: Ray3d, stack: &PaperStack) -> Option<Vec2> {
    let dir = ray.direction.as_vec3();
    let origin = ray.origin;
    let pt = stack.paper_thickness;
    let n = stack.layers.len();
    if n == 0 || dir.z.abs() < 1e-6 {
        return None;
    }
    for i in (0..n).rev() {
        let z = (i as f32 + 1.0) * pt;
        let t = (z - origin.z) / dir.z;
        if t < 0.0 {
            continue;
        }
        let p = origin + dir * t;
        let xy_f64 = geo::Point::new(p.x as f64, p.y as f64);
        if !stack.layers[i].contains(&xy_f64) {
            continue;
        }
        let above_has = i + 1 < n && stack.layers[i + 1].contains(&xy_f64);
        if !above_has {
            return Some(Vec2::new(p.x, p.y));
        }
    }
    None
}

// ─── Brush stroke ─────────────────────────────────────────────────

fn apply_brush_input(
    mouse: Res<ButtonInput<MouseButton>>,
    hover: Res<HoverPos>,
    brush: Res<Brush>,
    mut last: ResMut<StrokeLast>,
    mut stack: ResMut<PaperStack>,
    mut dirty: ResMut<StackDirty>,
) {
    if !mouse.pressed(MouseButton::Left) {
        last.0 = None;
        return;
    }
    let Some(pos) = hover.0 else { return };
    // Apply at most every `min_step` world units along the stroke,
    // so the cursor dragging doesn't dump 60 carves/sec on the same
    // area.
    let min_step = brush.radius * 0.4;
    if let Some(prev) = last.0 {
        if (prev - pos).length() < min_step {
            return;
        }
    }
    last.0 = Some(pos);
    carve_brush(&mut stack, &brush, pos);
    dirty.0 = true;
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

// ─── Debug panel ──────────────────────────────────────────────────

fn debug_panel(
    mut contexts: EguiContexts,
    debug: Res<DebugMode>,
    mut light: ResMut<LightControl>,
    mut camera: ResMut<CameraControl>,
    mut brush: ResMut<Brush>,
    mut stack: ResMut<PaperStack>,
    mut dirty: ResMut<StackDirty>,
    mut letters: ResMut<LettersUi>,
    mut doc: ResMut<DocumentUi>,
    mut form: ResMut<FormUi>,
    font: Res<LoadedFont>,
    font_lib: Res<FontLibrary>,
) -> Result {
    if !debug.0 {
        return Ok(());
    }
    let ctx = contexts.ctx_mut()?;

    egui::Window::new("Paper-stack editor")
        .default_pos([12.0, 12.0])
        .default_width(320.0)
        .show(ctx, |ui| {
            ui.collapsing("Tool", |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut brush.tool, Tool::Dig, "Dig (1)");
                    ui.selectable_value(&mut brush.tool, Tool::Extrude, "Extrude (2)");
                });
                ui.add(egui::Slider::new(&mut brush.radius, 0.05..=2.0).text("radius"));
                let mut s = brush.strength as i32;
                if ui.add(egui::Slider::new(&mut s, 1..=16).text("strength (layers)")).changed() {
                    brush.strength = s.max(1) as u16;
                }
                ui.add(
                    egui::Slider::new(&mut brush.layer_shrink, 0.5..=0.99)
                        .text("layer shrink")
                        .fixed_decimals(2),
                );
                if ui.button("Reset stack").clicked() {
                    stack.reset_full();
                    dirty.0 = true;
                }
            });

            ui.collapsing("Form fields (lines + UI shapes)", |ui| {
                let mut d = form.depth as i32;
                if ui
                    .add(egui::Slider::new(&mut d, 1..=N_LAYERS as i32).text("depth (layers)"))
                    .changed()
                {
                    form.depth = d.max(1) as u16;
                }
                ui.checkbox(&mut form.bevel, "chiseled bevel");
                if ui.button("Carve form").clicked() {
                    carve_form(&mut stack, form.depth, form.bevel);
                    dirty.0 = true;
                }
            });

            ui.collapsing("Letters", |ui| {
                let mut text = letters.text.clone();
                if ui.text_edit_singleline(&mut text).changed() {
                    letters.text = text;
                }
                ui.add(
                    egui::Slider::new(&mut letters.em_world, 0.4..=6.0)
                        .text("em (world)")
                        .fixed_decimals(2),
                );
                let mut d = letters.depth as i32;
                if ui
                    .add(egui::Slider::new(&mut d, 1..=N_LAYERS as i32).text("depth (layers)"))
                    .changed()
                {
                    letters.depth = d.max(1) as u16;
                }
                ui.checkbox(&mut letters.bevel, "chiseled bevel");
                if ui.button("Carve text into pad").clicked() {
                    carve_text(
                        &mut stack,
                        &font.0,
                        &letters.text,
                        letters.em_world,
                        letters.depth,
                        letters.bevel,
                    );
                    dirty.0 = true;
                }
            });

            ui.collapsing("Lorem ipsum (multi-font)", |ui| {
                ui.label(format!("{} fonts loaded", font_lib.fonts.len()));
                egui::ScrollArea::vertical().max_height(80.0).show(ui, |ui| {
                    for (i, f) in font_lib.fonts.iter().enumerate() {
                        ui.label(format!("  {i:>2}. {}", f.name));
                    }
                });
                ui.add(
                    egui::Slider::new(&mut doc.max_width, 2.0..=20.0)
                        .text("max width (world)"),
                );
                ui.add(
                    egui::Slider::new(&mut doc.line_factor, 1.0..=2.5)
                        .text("line spacing")
                        .fixed_decimals(2),
                );
                let mut d = doc.depth as i32;
                if ui
                    .add(egui::Slider::new(&mut d, 1..=N_LAYERS as i32).text("depth (layers)"))
                    .changed()
                {
                    doc.depth = d.max(1) as u16;
                }
                ui.checkbox(&mut doc.bevel, "chiseled bevel");
                ui.horizontal(|ui| {
                    if ui.button("Carve lorem ipsum").clicked() {
                        let lines = lorem_ipsum_doc(&font_lib, 4);
                        carve_document(
                            &mut stack,
                            &font_lib,
                            &lines,
                            doc.max_width,
                            doc.line_factor,
                            doc.depth,
                            doc.bevel,
                        );
                        dirty.0 = true;
                    }
                    if ui.button("Carve (8 words/line)").clicked() {
                        let lines = lorem_ipsum_doc(&font_lib, 8);
                        carve_document(
                            &mut stack,
                            &font_lib,
                            &lines,
                            doc.max_width,
                            doc.line_factor,
                            doc.depth,
                            doc.bevel,
                        );
                        dirty.0 = true;
                    }
                });
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
                        egui::Slider::new(&mut stack.pyramid_inset, 0.0..=0.05)
                            .text("pyramid inset (per-layer erosion)"),
                    )
                    .changed()
                {
                    dirty.0 = true;
                }
                ui.add(
                    egui::Slider::new(&mut stack.miter_limit, 1.0..=8.0).text("miter limit"),
                );
                ui.label(format!(
                    "stack height: {:.3}",
                    stack.layer_top_z(stack.n_layers().saturating_sub(1))
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
                    stack.palette = default_palette(N_LAYERS);
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
            ui.label("Tab to hide. Two-finger pan, pinch zoom, [ / ] tilt.");
        });
    Ok(())
}
