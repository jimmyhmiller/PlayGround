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
use bevy::camera::Viewport;
use bevy::camera::visibility::RenderLayers;
use bevy::image::{
    Image, ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor,
};
use bevy::input::ButtonInput;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::input::gestures::{PanGesture, PinchGesture};
use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::light::{CascadeShadowConfigBuilder, GlobalAmbientLight, NotShadowCaster};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
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

/// Polygon atlas produced by `extract_icons` — a list of icons each
/// decomposed into per-depth contour polygons in [0, 1]² icon-local
/// coordinates with y down.
#[derive(Resource, Deserialize, Debug)]
struct IconAtlas {
    legend: Vec<[u8; 3]>,
    icons: Vec<IconEntry>,
}

#[derive(Deserialize, Debug, Clone)]
struct IconEntry {
    row: usize,
    col: usize,
    name: String,
    aspect: f32,
    layers: Vec<IconAtlasLayer>,
}

#[derive(Deserialize, Debug, Clone)]
struct IconAtlasLayer {
    depth: u8,
    polygons: Vec<IconAtlasPolygon>,
}

#[derive(Deserialize, Debug, Clone)]
struct IconAtlasPolygon {
    exterior: Vec<[f32; 2]>,
    #[serde(default)]
    holes: Vec<Vec<[f32; 2]>>,
}

impl IconAtlas {
    fn get(&self, name: &str) -> Option<&IconEntry> {
        self.icons.iter().find(|i| i.name == name)
    }
}

fn load_icon_atlas() -> IconAtlas {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("assets/icons/icons.json");
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

/// Place an icon at world `center` with height `size_h`. Returns a
/// list of `(depth, polygons)` ready to be subtracted from the
/// stack: the icon's depth-K mask should be cut from the card's top
/// K layers (clamped to the card's actual thickness by the caller).
fn icon_at(
    atlas: &IconAtlas,
    name: &str,
    center: Vec2,
    size_h: f32,
) -> Vec<(u8, Vec<Polygon<f64>>)> {
    let Some(icon) = atlas.get(name) else {
        eprintln!("icon '{name}' not in atlas");
        return Vec::new();
    };
    // Atlas polys are y-down in [0,1]². Flip Y for our y-up world.
    let size_w = size_h * icon.aspect;
    let to_world = |[u, v]: [f32; 2]| -> Coord<f64> {
        Coord {
            x: (center.x + (u - 0.5) * size_w) as f64,
            y: (center.y - (v - 0.5) * size_h) as f64,
        }
    };
    let to_ring = |pts: &[[f32; 2]]| -> Option<LineString<f64>> {
        if pts.len() < 3 {
            return None;
        }
        let mut coords: Vec<Coord<f64>> = pts.iter().map(|p| to_world(*p)).collect();
        if coords[0] != *coords.last().unwrap() {
            coords.push(coords[0]);
        }
        Some(LineString::from(coords))
    };
    icon.layers
        .iter()
        .map(|layer| {
            let polys: Vec<Polygon<f64>> = layer
                .polygons
                .iter()
                .filter_map(|p| {
                    let ext = to_ring(&p.exterior)?;
                    let holes: Vec<LineString<f64>> =
                        p.holes.iter().filter_map(|h| to_ring(h)).collect();
                    Some(Polygon::new(ext, holes))
                })
                .collect();
            (layer.depth, polys)
        })
        .collect()
}
use ttf_parser::{Face, GlyphId, OutlineBuilder};

// World-unit dimensions of the paper.
const PAPER_W: f32 = 16.0;
const PAPER_H: f32 = 16.0;
const N_LAYERS: u16 = 32;
const DEFAULT_PAPER_THICKNESS: f32 = 0.018;
/// Per-layer inward inset of the carve outline. Stacking N
/// successively-eroded carves at this inset gives a chiseled bevel
/// whose slope is `pyramid_inset / paper_thickness`.
const DEFAULT_PYRAMID_INSET: f32 = 0.012;
const DEFAULT_MITER_LIMIT: f32 = 4.0;
/// How many times the paper texture repeats across the full PAPER_W
/// span. Smaller = more detail per repeat; larger = finer grain. The
/// paper texture is fairly uniform so 1.5 keeps obvious tiling out of
/// sight at typical zoom levels.
const PAPER_TEX_TILES: f32 = 1.5;

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
const CURRENT_STATE_VERSION: u32 = 7;

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
    #[serde(default)]
    chamfer_w_frac: f32,
    #[serde(default)]
    chamfer_h_frac: f32,
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
    /// Whether this slot casts shadows. Older saved states are
    /// missing this field; serde_default keeps them loadable, and
    /// the version-aware loader fills the right initial value.
    #[serde(default)]
    casts_shadow: bool,
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
    /// Whether this slot's directional light casts shadows.
    /// Multiple shadow casters tend to produce muddy double-shadows
    /// on the layered paper, so the default scene only gives this
    /// to the key light — but it's a per-slot toggle now.
    casts_shadow: bool,
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

// ─── Debug 3D scene view ──────────────────────────────────────────

const SCENE_LAYER: usize = 1;
/// Radius of the dome on which light gizmos are placed. Light
/// directions are unit vectors; we draw the gizmo at `dir * DOME_R`
/// so the user has a tangible 3D position to grab.
const LIGHT_GIZMO_DOME_R: f32 = 8.0;
const LIGHT_GIZMO_PICK_R: f32 = 0.55;

#[derive(Component)]
struct MainCam;

#[derive(Component)]
struct SceneCam;

#[derive(Component)]
struct LightGizmo(usize);

#[derive(Component)]
struct LightGizmoMaterial(Handle<StandardMaterial>);

#[derive(Resource)]
struct DebugView {
    enabled: bool,
    /// Orbit yaw (around world Y, radians)
    yaw: f32,
    /// Orbit pitch (radians, +up)
    pitch: f32,
    /// Camera distance from origin
    dist: f32,
    selected_light: Option<usize>,
    dragging_light: bool,
    orbiting: bool,
}

impl Default for DebugView {
    fn default() -> Self {
        Self {
            enabled: false,
            yaw: 0.7,
            pitch: 0.6,
            dist: 22.0,
            selected_light: None,
            dragging_light: false,
            orbiting: false,
        }
    }
}

#[derive(Resource)]
struct PaperStack {
    paper_size: Vec2,
    /// 2D region of each layer; layer index is identity-only — z is
    /// explicit via `layer_z` so multiple layers can share a height
    /// in different XY territories.
    layers: Vec<MultiPolygon<f64>>,
    /// World-z of the TOP face of each layer. Bottom face is
    /// `layer_z[i] - paper_thickness`. Defaults to `(i+1) * pt` to
    /// keep the original carved-down model working unchanged, but
    /// callers can override with arbitrary values.
    layer_z: Vec<f32>,
    /// Per-layer wall slope: how far the BOTTOM of the wall is
    /// offset INTO THE VOID from the top, in world units.
    layer_slope_inset: Vec<f32>,
    palette: Vec<Color>,
    paper_thickness: f32,
    pyramid_inset: f32,
    miter_limit: f32,
    /// Top-of-wall chamfer width as a fraction of the wall's
    /// `slope_inset`. 0 = no chamfer; 0.5 = chamfer takes half the
    /// slope. The chamfer faces up-and-out and catches the key/fill
    /// light, producing the thin "shine" along every cut edge.
    chamfer_w_frac: f32,
    /// Top-of-wall chamfer height as a fraction of `paper_thickness`.
    chamfer_h_frac: f32,
}

impl PaperStack {
    fn new(size: Vec2, n_layers: u16, palette: Vec<Color>) -> Self {
        let full = full_rect(size);
        let pt = DEFAULT_PAPER_THICKNESS;
        Self {
            paper_size: size,
            layers: vec![full; n_layers as usize],
            layer_z: (0..n_layers as usize).map(|i| (i as f32 + 1.0) * pt).collect(),
            layer_slope_inset: vec![DEFAULT_PYRAMID_INSET; n_layers as usize],
            palette,
            paper_thickness: pt,
            pyramid_inset: DEFAULT_PYRAMID_INSET,
            miter_limit: DEFAULT_MITER_LIMIT,
            chamfer_w_frac: 0.5,
            chamfer_h_frac: 0.2,
        }
    }
    fn n_layers(&self) -> u16 {
        self.layers.len() as u16
    }
    fn reset_full(&mut self) {
        let full = full_rect(self.paper_size);
        let pt = self.paper_thickness;
        for layer in self.layers.iter_mut() {
            *layer = full.clone();
        }
        for (i, z) in self.layer_z.iter_mut().enumerate() {
            *z = (i as f32 + 1.0) * pt;
        }
        for s in self.layer_slope_inset.iter_mut() {
            *s = self.pyramid_inset;
        }
    }
    /// World-z of the top of layer `i`.
    fn layer_top_z(&self, i: u16) -> f32 {
        self.layer_z
            .get(i as usize)
            .copied()
            .unwrap_or((i as f32 + 1.0) * self.paper_thickness)
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
        // Key — soft top-left, dominant. Casts shadows.
        LightDef { dir: Vec3::new(-0.30, -0.30, 0.90), intensity: 0.40, color_idx: 0, casts_shadow: true },
        // Fill — gentler top-right to lift shadows on the opposite side.
        LightDef { dir: Vec3::new( 0.40, -0.20, 0.85), intensity: 0.10, color_idx: 0, casts_shadow: false },
        // Back rim — faint, opposite direction.
        LightDef { dir: Vec3::new( 0.0,   0.45, 0.70), intensity: 0.05, color_idx: 0, casts_shadow: false },
    ]
}

/// Hue-rotated rainbow palette used for layer side-banding.
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
    let font_data = load_font();
    let font_library = load_font_library();
    let icon_atlas = load_icon_atlas();
    let screenshot_out = std::env::var("SCREENSHOT_OUT").ok();
    // BOOK_UI_TUNE=1 keeps the app running with a hidden window and
    // the demo scene pre-built — used by the python tuning harness
    // to drive the app via /tmp/book_ui_tune.json without needing
    // a fresh build/launch per iteration.
    let tune_mode = std::env::var("BOOK_UI_TUNE").is_ok();
    let auto_scene = screenshot_out.is_some() || tune_mode;
    // In screenshot / tune mode skip persisted state — we want a
    // deterministic baseline (default lights / ambient / camera) so
    // automated palette comparisons don't drift with whatever the
    // user happened to leave on disk.
    let persisted = if auto_scene { None } else { load_persisted_state() };

    // Always start framed on the demo scene. The user can pan/zoom/tilt
    // after; that gets persisted as usual.
    let camera_ctrl = CameraControl { pan: Vec2::ZERO, zoom: 0.48, tilt: 0.0 };

    let light_ctrl = LightControl {
        lights: persisted
            .as_ref()
            .filter(|s| s.version == CURRENT_STATE_VERSION && s.lights.len() == 3)
            .map(|s| {
                let mut arr = default_lights();
                for (i, pl) in s.lights.iter().take(3).enumerate() {
                    arr[i] = LightDef {
                        dir: Vec3::from_array(pl.dir),
                        intensity: pl.intensity,
                        color_idx: pl.color_idx,
                        casts_shadow: pl.casts_shadow,
                    };
                }
                arr
            })
            .unwrap_or_else(default_lights),
        ambient: persisted
            .as_ref()
            .filter(|s| s.version == CURRENT_STATE_VERSION)
            .map(|s| s.ambient)
            .unwrap_or(0.10),
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
        if s.version == CURRENT_STATE_VERSION {
            stack.chamfer_w_frac = s.chamfer_w_frac;
            stack.chamfer_h_frac = s.chamfer_h_frac;
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

    let default_brush = Brush {
        tool: Tool::Dig,
        radius: 0.05,
        strength: 4,
        layer_shrink: 0.9,
    };
    let brush = persisted
        .as_ref()
        .filter(|s| s.version == CURRENT_STATE_VERSION)
        .map(|s| Brush {
            tool: Tool::Dig,
            radius: s.brush_radius.clamp(0.002, 1.0),
            strength: s.brush_strength.max(1),
            layer_shrink: s.brush_layer_shrink.clamp(0.5, 0.99),
        })
        .unwrap_or(default_brush);

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

    let window_resolution = if auto_scene {
        (1200u32, 1200u32).into()
    } else {
        (1400u32, 900u32).into()
    };
    // Screenshot mode: open the window unfocused and hidden so it
    // doesn't steal focus from whatever the user is doing. The
    // renderer still draws to it; we capture and exit.
    let primary_window = Window {
        title: "paper-stack editor".into(),
        resolution: window_resolution,
        visible: !auto_scene,
        focused: !auto_scene,
        ..default()
    };

    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins.set(WindowPlugin {
            primary_window: Some(primary_window),
            ..default()
        }),
    )
        .add_plugins(EguiPlugin::default())
        .insert_resource(DebugMode(false))
        .insert_resource(DebugView::default())
        .insert_resource(TuningWatcher::default())
        .insert_resource(TuneMode(tune_mode))
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
        .insert_resource(icon_atlas)
        .insert_resource(letters)
        .insert_resource(doc)
        .insert_resource(FormUi { depth: 2, bevel: false })
        .insert_resource(ClearColor(Color::srgb(0.05, 0.04, 0.07)))
        // 8192-sq directional shadow map. Combined with the single
        // tightly-fit cascade above, this gives the best texel
        // density we can practically afford for the wall→shelf
        // shadows on a 0.018-thick layer stack. ~256 MB of GPU
        // memory for the shadow target.
        .insert_resource(bevy::light::DirectionalLightShadowMap { size: 8192 })
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                toggle_debug,
                toggle_debug_view,
                debug_inputs,
                scene_view_input,
                update_hover,
                apply_brush_input,
                regen_stack_mesh,
                sync_lights,
                sync_ambient,
                sync_camera,
                sync_scene_cam,
                sync_light_gizmos,
                apply_viewport_layout,
                poll_tuning_file,
                save_state_periodic,
            ),
        )
        .add_systems(EguiPrimaryContextPass, debug_panel)
        // Always start with the Create Account scene already carved.
        .add_systems(Startup, auto_build_create_account_scene.after(setup));

    if let Some(out_path) = screenshot_out {
        app.insert_resource(AutoScreenshot {
            path: out_path,
            state: AutoScreenshotState::Waiting(60),
        });
        app.add_systems(Update, auto_screenshot_system);
    }

    app.run();
}

/// Startup hook used in screenshot mode: re-create the reference
/// "Create Account" scene into the freshly-initialized stack.
fn auto_build_create_account_scene(
    mut stack: ResMut<PaperStack>,
    font_lib: Res<FontLibrary>,
    icon_atlas: Res<IconAtlas>,
    mut dirty: ResMut<StackDirty>,
) {
    stack.reset_full();
    carve_create_account_form(&mut stack, &font_lib, &icon_atlas);
    dirty.0 = true;
}

// ─── Live tuning over a JSON command file ─────────────────────────
//
// The app polls `/tmp/book_ui_tune.json` ~twice a second. Whenever
// the file's mtime changes, we deserialize a `TuneCmd` and apply the
// fields that are present:
//   { "palette":       [[r,g,b], ...]  // sRGB floats, optional
//   , "ambient":       0.10             // optional
//   , "light_intensities": [0.4, 0.1, 0.05]  // optional, len ≤ 3
//   , "screenshot_to": "/tmp/iter5.png" }   // optional one-shot
//
// This lets a python harness nudge palette / lighting and snap a
// screenshot for measurement without ever restarting the app.

#[derive(serde::Deserialize, Default)]
struct TuneCmd {
    /// Monotonic sequence number — the harness bumps this on every
    /// command so we can tell back-to-back writes apart even when
    /// the filesystem's mtime resolution can't.
    #[serde(default)]
    seq: u64,
    #[serde(default)]
    palette: Option<Vec<[f32; 3]>>,
    #[serde(default)]
    ambient: Option<f32>,
    #[serde(default)]
    light_intensities: Option<Vec<f32>>,
    #[serde(default)]
    screenshot_to: Option<String>,
}

#[derive(Resource, Default)]
struct TuningWatcher {
    last_seq: u64,
    poll_counter: u32,
    /// When a tune command asks for a screenshot, we don't spawn it
    /// immediately — the mesh regen runs in the SAME Update tick
    /// and Bevy doesn't guarantee an ordering between unrelated
    /// systems, so the screenshot can be taken before the new
    /// vertex colors land on the GPU. Defer by a few frames.
    pending_screenshot: Option<(u32, String)>,
}

const TUNE_PATH: &str = "/tmp/book_ui_tune.json";
const TUNE_DONE_PATH: &str = "/tmp/book_ui_tune.done";

#[derive(Resource)]
struct TuneMode(bool);

fn poll_tuning_file(
    mut commands: Commands,
    tune_mode: Res<TuneMode>,
    mut watcher: ResMut<TuningWatcher>,
    mut stack: ResMut<PaperStack>,
    mut light: ResMut<LightControl>,
    mut dirty: ResMut<StackDirty>,
) {
    // Tune file only drives the app when BOOK_UI_TUNE=1. Otherwise a
    // stale /tmp/book_ui_tune.json from an old harness run would
    // silently overwrite the user's saved lights/palette every poll.
    if !tune_mode.0 {
        return;
    }
    watcher.poll_counter = watcher.poll_counter.wrapping_add(1);

    // Service any deferred screenshot first (every frame). We wait
    // a few frames after applying a tune so the mesh regen lands
    // on the GPU before the screenshot is captured.
    if let Some((frames_left, path)) = watcher.pending_screenshot.take() {
        if frames_left == 0 {
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk(path.clone()));
            // The screenshot writer is async — give it ~0.5s on a
            // background thread before announcing the done file.
            let done = TUNE_DONE_PATH.to_string();
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(500));
                let _ = std::fs::write(&done, path.as_bytes());
            });
        } else {
            watcher.pending_screenshot = Some((frames_left - 1, path));
        }
    }

    if watcher.poll_counter % 10 != 0 {
        return;
    }
    let Ok(bytes) = std::fs::read(TUNE_PATH) else { return; };
    let cmd: TuneCmd = match serde_json::from_slice(&bytes) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[tune] bad json: {e}");
            return;
        }
    };
    if cmd.seq == 0 || cmd.seq == watcher.last_seq {
        return;
    }
    watcher.last_seq = cmd.seq;

    if let Some(palette) = cmd.palette {
        for (i, c) in palette.iter().enumerate() {
            if i < stack.palette.len() {
                stack.palette[i] = Color::srgb(c[0], c[1], c[2]);
            }
        }
        dirty.0 = true;
    }
    if let Some(amb) = cmd.ambient {
        light.ambient = amb;
    }
    if let Some(intensities) = cmd.light_intensities {
        for (i, v) in intensities.iter().take(3).enumerate() {
            light.lights[i].intensity = *v;
        }
    }
    if let Some(out) = cmd.screenshot_to {
        // Defer 4 frames — enough for the dirty-flag-driven mesh
        // regen + GPU upload to complete before the screenshot is
        // captured. Independent of system ordering.
        watcher.pending_screenshot = Some((4, out));
    }
    eprintln!("[tune] applied seq={}", cmd.seq);
}

fn save_state_periodic(
    auto: Option<Res<AutoScreenshot>>,
    time: Res<Time>,
    mut last_save: Local<f64>,
    camera: Res<CameraControl>,
    light: Res<LightControl>,
    brush: Res<Brush>,
    stack: Res<PaperStack>,
    letters: Res<LettersUi>,
    doc: Res<DocumentUi>,
) {
    // Don't persist in screenshot mode — it'd overwrite the user's
    // saved palette and brush settings with the demo scene's values.
    if auto.is_some() {
        return;
    }
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
                casts_shadow: l.casts_shadow,
            })
            .collect(),
        ambient: light.ambient,
        paper_thickness: stack.paper_thickness,
        pyramid_inset: stack.pyramid_inset,
        miter_limit: stack.miter_limit,
        chamfer_w_frac: stack.chamfer_w_frac,
        chamfer_h_frac: stack.chamfer_h_frac,
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
    mut images: ResMut<Assets<Image>>,
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
        // Gaussian PCF blurs the shadow map sampling so individual
        // texel boundaries don't show as ridges along faceted wall
        // edges.
        bevy::light::ShadowFilteringMethod::Gaussian,
        MainCam,
    ));

    // Second camera for the debug "scene view" — orbits the paper
    // stack and renders the light gizmos (RenderLayer 1). Inactive by
    // default; enabled when `DebugView.enabled` is on.
    commands.spawn((
        Camera3d::default(),
        Camera {
            is_active: false,
            order: 1,
            ..default()
        },
        Transform::from_xyz(12.0, 12.0, 18.0).looking_at(Vec3::ZERO, Vec3::Z),
        Projection::from(PerspectiveProjection {
            fov: 35f32.to_radians(),
            near: 0.1,
            far: 400.0,
            ..default()
        }),
        RenderLayers::from_layers(&[0, SCENE_LAYER]),
        SceneCam,
    ));

    // Light gizmos: one unlit emissive sphere per directional light,
    // visible only on render layer 1 (so it shows in the scene-view
    // camera but not in the main book UI).
    let gizmo_mesh = meshes.add(Sphere::new(0.45));
    for i in 0..3 {
        let mat = materials.add(StandardMaterial {
            base_color: Color::WHITE,
            emissive: LinearRgba::WHITE,
            unlit: true,
            ..default()
        });
        commands.spawn((
            Mesh3d(gizmo_mesh.clone()),
            MeshMaterial3d(mat.clone()),
            Transform::from_xyz(0.0, 0.0, LIGHT_GIZMO_DOME_R),
            RenderLayers::layer(SCENE_LAYER),
            NotShadowCaster,
            LightGizmo(i),
            LightGizmoMaterial(mat),
        ));
    }

    // Faint reference plane at the paper level — just for the scene
    // cam, so the user has spatial context for where the lights sit
    // relative to the paper.
    // Thin reference plate (lying flat in the XY plane, just below
    // the paper) — gives the scene cam a sense of where the book
    // surface is relative to the lights.
    let ref_mesh = meshes.add(Cuboid::new(PAPER_W * 1.05, PAPER_H * 1.05, 0.005));
    let ref_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.4, 0.4, 0.45, 0.35),
        emissive: LinearRgba::new(0.04, 0.04, 0.06, 1.0),
        unlit: true,
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    commands.spawn((
        Mesh3d(ref_mesh),
        MeshMaterial3d(ref_mat),
        Transform::from_xyz(0.0, 0.0, -0.01),
        RenderLayers::layer(SCENE_LAYER),
        NotShadowCaster,
    ));

    for i in 0..3 {
        commands.spawn((
            DirectionalLight {
                color: Color::WHITE,
                illuminance: 0.0,
                shadows_enabled: false,
                // Bumped above Bevy's defaults for thin stacked
                // geometry: at 0.018-unit layer thickness, undersized
                // bias causes wavy shadow acne along every transition.
                shadow_depth_bias: 0.04,
                shadow_normal_bias: 0.8,
                ..default()
            },
            CascadeShadowConfigBuilder {
                // Single cascade tightly fit to the paper. With the
                // 8192 shadow-map resolution below, this gives
                // ~270 texels per world unit (≈ 5 texels per layer
                // thickness) — enough that PCF can smooth the wall
                // shadows on each shelf instead of producing a wavy
                // few-texel-wide ridge.
                num_cascades: 1,
                minimum_distance: 0.05,
                first_cascade_far_bound: 30.0,
                maximum_distance: 30.0,
                ..default()
            }
            .build(),
            DirLightIndex(i),
        ));
    }

    // Paper textures, loaded directly with a CPU-generated mipmap
    // chain. Bevy 0.18's JPG loader produces a single-mip texture;
    // viewed at high tilt, that aliases into vertical streaks because
    // the GPU has nothing smaller to sample from. Mipmaps + 16x
    // anisotropy collapse the aliasing.
    let diffuse_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("assets/textures/paper_diffuse.jpg");
    let normal_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("assets/textures/paper_normal.jpg");
    let paper_diffuse = images.add(load_mipped_image(&diffuse_path, true));
    // Normal map is linear data — sRGB-decoding the encoded
    // (R, G, B) = (Tx, Ty, Tz) values would warp the tangent vectors.
    let paper_normal = images.add(load_mipped_image(&normal_path, false));

    let stack_mat = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        base_color_texture: Some(paper_diffuse),
        normal_map_texture: Some(paper_normal),
        perceptual_roughness: 1.0,
        metallic: 0.0,
        reflectance: 0.0,
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
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut colors: Vec<[f32; 4]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let pt = stack.paper_thickness;
    let miter = stack.miter_limit as f64;
    let n = stack.layers.len();

    for i in 0..n {
        let layer = &stack.layers[i];
        if layer.0.is_empty() {
            continue;
        }
        let z_top = stack.layer_top_z(i as u16);
        let z_bot = z_top - pt;
        let color = color_to_rgba(stack.palette[i % stack.palette.len()]);

        // Exposed top = layer's region MINUS the union of every
        // layer that sits at a strictly higher z. Layers at the
        // same z but in different XY don't occlude each other.
        let mut above_union: MultiPolygon<f64> = MultiPolygon(vec![]);
        for j in 0..n {
            if j == i {
                continue;
            }
            if stack.layer_top_z(j as u16) > z_top + 1e-6
                && !stack.layers[j].0.is_empty()
            {
                above_union = above_union.union(&stack.layers[j]);
            }
        }
        let exposed = if above_union.0.is_empty() {
            layer.clone()
        } else {
            layer.difference(&above_union)
        };

        if !exposed.0.is_empty() {
            tessellate_region(
                &exposed,
                z_top,
                color,
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut colors,
                &mut indices,
            );
        }

        // Walls along EVERY boundary of this layer (exterior +
        // interior holes). Each wall is one paper_thickness tall.
        let slope = stack
            .layer_slope_inset
            .get(i)
            .copied()
            .unwrap_or(stack.pyramid_inset) as f64;
        for poly in &layer.0 {
            emit_wall_for_ring(
                poly.exterior(),
                z_top,
                z_bot,
                slope,
                miter,
                stack.chamfer_w_frac,
                stack.chamfer_h_frac,
                color,
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut colors,
                &mut indices,
            );
            for hole in poly.interiors() {
                emit_wall_for_ring(
                    hole,
                    z_top,
                    z_bot,
                    slope,
                    miter,
                    stack.chamfer_w_frac,
                    stack.chamfer_h_frac,
                    color,
                    &mut positions,
                    &mut normals,
                    &mut uvs,
                    &mut colors,
                    &mut indices,
                );
            }
        }
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
    // Tangents are required for the normal-map sampling on the
    // StandardMaterial. generate_tangents needs positions, normals,
    // UVs, and indices already present (they are).
    let _ = mesh.generate_tangents();
    mesh
}

fn tessellate_region(
    region: &MultiPolygon<f64>,
    z: f32,
    color: [f32; 4],
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
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
            uvs.push(planar_uv(v[0], v[1]));
            colors.push(tinted(color, v[0], v[1]));
        }
        // Lyon emits triangles in y-down convention. When we promote
        // them to a 3D plane with y_3D = y_2D and z = const, the
        // geometric winding flips relative to a +Z viewer — the cap
        // would render as a back face and get culled. Swap each
        // triangle's last two indices to restore CCW-from-+Z.
        for chunk in buffers.indices.chunks_exact(3) {
            indices.push(base + chunk[0]);
            indices.push(base + chunk[2]);
            indices.push(base + chunk[1]);
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

/// Emit a wall along a closed ring with **smoothly averaged per-vertex
/// normals** so curved sections (circles, rounded-rect corners) shade
/// as continuous slopes instead of flat facets. The wall's BOTTOM is
/// offset inward by `slope_inset`; `slope_inset = 0` ⇒ vertical wall.
fn emit_wall_for_ring(
    ring: &LineString<f64>,
    z_top: f32,
    z_bot: f32,
    slope_inset: f64,
    miter_limit: f64,
    chamfer_w_frac: f32,
    chamfer_h_frac: f32,
    color: [f32; 4],
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    let pts_top = ring_points(ring);
    let n = pts_top.len();
    if n < 3 {
        return;
    }
    let coords = ring_to_coords(ring);
    let pts_bot: Vec<Vec2> = if slope_inset > 1e-9 {
        miter_offset_right_f64(&coords, slope_inset, miter_limit)
            .iter()
            .map(|c| Vec2::new(c.x as f32, c.y as f32))
            .collect()
    } else {
        pts_top.clone()
    };

    // Top chamfer: a thin up-and-out facing ring inserted between
    // the cap rim and the wall proper. The wall faces sideways and
    // sits in shadow under the (mostly downward) lights; the chamfer
    // tilts upward enough to catch direct light, producing a thin
    // shine highlight along every cut edge.
    //
    // chamfer_w is taken from slope_inset so the chamfer never
    // overhangs further than the wall already slopes — on vertical
    // walls (slope_inset == 0) we suppress it entirely, since an
    // outward-leaning lip would look wrong.
    // Chamfer dimensions are scaled off the wall *thickness*, not
    // its slope — the demo scene sets per-layer slope_inset to 0 for
    // every wall, so basing chamfer_w on slope produced zero chamfer
    // even at max slider value. With a vertical wall (slope == 0)
    // the resulting chamfer overhangs the wall slightly, which is
    // fine: it still reads as a beveled top edge catching light.
    let pt = (z_top - z_bot).max(0.0);
    let chamfer_w = pt * chamfer_w_frac;
    let chamfer_h = pt * chamfer_h_frac;
    let has_chamfer = chamfer_w > 1e-6 && chamfer_h > 1e-6;
    let pts_chamfer: Vec<Vec2> = if has_chamfer {
        miter_offset_right_f64(&coords, chamfer_w as f64, miter_limit)
            .iter()
            .map(|c| Vec2::new(c.x as f32, c.y as f32))
            .collect()
    } else {
        pts_top.clone()
    };
    let z_chamfer = if has_chamfer { z_top - chamfer_h } else { z_top };

    // Cumulative arclength along the top rim, used as the U axis for
    // wall UVs. Without this — i.e. if walls reused the cap's planar
    // (x, y) UV — the top and bottom rim would share UVs (vertical
    // walls) or differ by only `slope_inset`, giving a near-degenerate
    // UV quad that the sampler stretches across the whole wall face
    // as a single column of texels.
    //
    // V (vertical on the wall face) deliberately uses a different
    // scale than U: the wall is only paper_thickness (~0.018) tall,
    // so a strict world-units mapping samples just a few texels of
    // texture height — which the GPU smears as vertical streaks
    // across the visible wall. Mapping each wall face to a fixed
    // fraction of the texture instead gives a real chunk of grain on
    // every wall regardless of its physical thinness.
    let mut cum_len = vec![0.0_f32; n + 1];
    for i in 0..n {
        let j = (i + 1) % n;
        cum_len[i + 1] = cum_len[i] + (pts_top[j] - pts_top[i]).length();
    }
    let u_scale = PAPER_TEX_TILES / PAPER_W;
    // Per-wall vertical span in UV. 0.15 = ~15% of texture height;
    // big enough that the wall shows recognizable paper grain but
    // small enough that it reads as continuous fibers rather than a
    // distinct second tile. The chamfer takes a small slice at the
    // top of that span so its texture flows continuously into the
    // wall below.
    const WALL_V_SPAN: f32 = 0.15;
    const CHAMFER_V_FRAC: f32 = 0.2;
    let v_cham_top = 0.0;
    let v_cham_bot = WALL_V_SPAN * CHAMFER_V_FRAC;
    let v_top = if has_chamfer { v_cham_bot } else { 0.0 };
    let v_bot = WALL_V_SPAN;

    // Per-quad FLAT normals. Smooth/averaged per-vertex normals
    // looked nicer on continuous curves but caused shadow ridge
    // artifacts: the shading pretends the wall is curved, but the
    // shadow caster is still faceted, so each segment-vertex
    // produced a tiny shadow discontinuity. Flat shading matches
    // the geometry exactly — uniform shadows per facet.

    // Chamfer ring (when present): one quad per segment, going from
    // the cap rim down-and-inward to the wall's true top.
    if has_chamfer {
        for i in 0..n {
            let j = (i + 1) % n;
            let p_top_i = Vec3::new(pts_top[i].x, pts_top[i].y, z_top);
            let p_top_j = Vec3::new(pts_top[j].x, pts_top[j].y, z_top);
            let p_cham_i = Vec3::new(pts_chamfer[i].x, pts_chamfer[i].y, z_chamfer);
            let p_cham_j = Vec3::new(pts_chamfer[j].x, pts_chamfer[j].y, z_chamfer);

            let edge_top = p_top_j - p_top_i;
            let down = p_cham_i - p_top_i;
            let n_face = down
                .cross(edge_top)
                .try_normalize()
                .unwrap_or(Vec3::Z);
            let normal = [n_face.x, n_face.y, n_face.z];

            let u_i = cum_len[i] * u_scale;
            let u_j = cum_len[i + 1] * u_scale;
            let cham_uvs = [
                [u_i, v_cham_top],
                [u_i, v_cham_bot],
                [u_j, v_cham_bot],
                [u_j, v_cham_top],
            ];

            let base = positions.len() as u32;
            for (k, v) in [p_top_i, p_cham_i, p_cham_j, p_top_j].iter().enumerate() {
                positions.push([v.x, v.y, v.z]);
                normals.push(normal);
                uvs.push(cham_uvs[k]);
                colors.push(tinted(color, v.x, v.y));
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }

    // Main wall: from chamfer-bottom (or cap rim, if no chamfer)
    // down to z_bot.
    let pts_wtop = if has_chamfer { &pts_chamfer } else { &pts_top };
    for i in 0..n {
        let j = (i + 1) % n;
        let p_top_i = Vec3::new(pts_wtop[i].x, pts_wtop[i].y, z_chamfer);
        let p_top_j = Vec3::new(pts_wtop[j].x, pts_wtop[j].y, z_chamfer);
        let p_bot_i = Vec3::new(pts_bot[i].x, pts_bot[i].y, z_bot);
        let p_bot_j = Vec3::new(pts_bot[j].x, pts_bot[j].y, z_bot);

        let edge_top = p_top_j - p_top_i;
        let down = p_bot_i - p_top_i;
        let n_face = down
            .cross(edge_top)
            .try_normalize()
            .unwrap_or(Vec3::Z);
        let normal = [n_face.x, n_face.y, n_face.z];

        let u_i = cum_len[i] * u_scale;
        let u_j = cum_len[i + 1] * u_scale;
        let wall_uvs = [
            [u_i, v_top],
            [u_i, v_bot],
            [u_j, v_bot],
            [u_j, v_top],
        ];

        let base = positions.len() as u32;
        for (k, v) in [p_top_i, p_bot_i, p_bot_j, p_top_j].iter().enumerate() {
            positions.push([v.x, v.y, v.z]);
            normals.push(normal);
            uvs.push(wall_uvs[k]);
            colors.push(tinted(color, v.x, v.y));
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

/// Decode a JPG and build a Bevy `Image` with a full mipmap chain
/// (full-res down to 1×1 by box filter), wrapping repeat with 16x
/// anisotropy. Mipmaps are essential — at high camera tilt the cap
/// textures get sampled at oblique angles, and a single-mip texture
/// aliases into pronounced vertical streaks.
fn load_mipped_image(path: &std::path::Path, is_srgb: bool) -> Image {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
        .to_rgba8();

    // Square + power-of-two so each mip halves cleanly. Round DOWN
    // to the nearest power of two — `next_power_of_two` returns the
    // input itself when already PoT, but this needs to land on the
    // largest PoT ≤ min(w, h) regardless.
    let raw = img.width().min(img.height());
    let dim = if raw.is_power_of_two() {
        raw
    } else {
        raw.next_power_of_two() / 2
    };
    let mut current = if img.width() == dim && img.height() == dim {
        img
    } else {
        image::imageops::resize(&img, dim, dim, image::imageops::FilterType::Triangle)
    };

    let mut data: Vec<u8> = current.as_raw().clone();
    let mut size = dim;
    let mut mip_count: u32 = 1;
    while size > 1 {
        let next = size / 2;
        current = image::imageops::resize(&current, next, next, image::imageops::FilterType::Triangle);
        data.extend_from_slice(current.as_raw());
        size = next;
        mip_count += 1;
    }

    let format = if is_srgb {
        TextureFormat::Rgba8UnormSrgb
    } else {
        TextureFormat::Rgba8Unorm
    };
    // `Image::new` asserts that data length matches the *base level*
    // volume only — it doesn't know about mip chains and would panic
    // on the concatenated data. `new_uninit` skips that check; we
    // attach the full mip data and bump mip_level_count manually.
    let mut image = Image::new_uninit(
        Extent3d { width: dim, height: dim, depth_or_array_layers: 1 },
        TextureDimension::D2,
        format,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.data = Some(data);
    image.texture_descriptor.mip_level_count = mip_count;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        address_mode_w: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        anisotropy_clamp: 16,
        ..ImageSamplerDescriptor::linear()
    });
    image
}

/// Planar UV from world XY for cap (top) faces. Wall faces compute
/// their own UVs from arclength + z — see `emit_wall_for_ring`.
fn planar_uv(x: f32, y: f32) -> [f32; 2] {
    [
        x * PAPER_TEX_TILES / PAPER_W + 0.5,
        y * PAPER_TEX_TILES / PAPER_H + 0.5,
    ]
}

fn color_to_rgba(c: Color) -> [f32; 4] {
    let lc = c.to_linear();
    [lc.red, lc.green, lc.blue, lc.alpha]
}

/// Identity. Per-vertex tinting interpolates linearly across long
/// fan-triangles in lyon's tessellation of rect-with-hole regions,
/// producing visible diagonal streaks radiating from the outer
/// corners. Paper texture is sampled in fragment space via the
/// diffuse + normal maps on the StandardMaterial, not via
/// per-vertex modulation.
fn tinted(color: [f32; 4], _x: f32, _y: f32) -> [f32; 4] {
    color
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
    if n == 0 {
        return;
    }
    let shrink = brush.layer_shrink.clamp(0.05, 0.99);
    let mut shapes: Vec<Option<MultiPolygon<f64>>> = Vec::with_capacity(brush.strength as usize);
    for k in 0..brush.strength as usize {
        let r = brush.radius * shrink.powi(k as i32);
        if r < 0.002 {
            break;
        }
        let circ = circle_polygon(world_pos, r, 48);
        shapes.push(Some(MultiPolygon(vec![circ])));
    }

    match brush.tool {
        Tool::Dig => {
            // Start from the topmost layer that still has material at
            // the brush center, so dragging into an already-dug pit
            // keeps cutting deeper instead of no-op'ing on absent layers.
            let pt_geo = geo::Point::new(world_pos.x as f64, world_pos.y as f64);
            let mut start: Option<usize> = None;
            for i in (0..n).rev() {
                if stack.layers[i].contains(&pt_geo) {
                    start = Some(i);
                    break;
                }
            }
            let Some(start) = start else { return };
            carve_stepped_from(stack, &shapes, start);
        }
        Tool::Extrude => extrude_stepped(stack, &shapes),
    }
}

/// Subtract `shapes[k]` from layer `start - k` (clamped to 0). Used by
/// the brush to carve down from a chosen surface layer rather than
/// always from the absolute top.
fn carve_stepped_from(
    stack: &mut PaperStack,
    shapes: &[Option<MultiPolygon<f64>>],
    start: usize,
) {
    for (k, maybe_shape) in shapes.iter().enumerate() {
        if k > start {
            break;
        }
        if let Some(shape) = maybe_shape {
            if shape.0.is_empty() {
                continue;
            }
            let layer_idx = start - k;
            stack.layers[layer_idx] = stack.layers[layer_idx].difference(shape);
        }
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

/// Mirror of `miter_offset_left_f64` that offsets to the *right* of
/// the walking direction. Used for sloped wall bottoms — for both
/// CCW exteriors and CW interior holes the wall slopes into the
/// "void" side, which is to the right of the walking direction.
fn miter_offset_right_f64(pts: &[Coord<f64>], inset: f64, miter_limit: f64) -> Vec<Coord<f64>> {
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
        // Right perpendicular of (dx, dy) = (dy, -dx).
        let n_in = (e_in.1, -e_in.0);
        let n_out = (e_out.1, -e_out.0);
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

fn rect_pts_ccw(cx: f32, cy: f32, w: f32, h: f32) -> Vec<(f64, f64)> {
    let hw = w as f64 * 0.5;
    let hh = h as f64 * 0.5;
    let cx = cx as f64;
    let cy = cy as f64;
    vec![
        (cx - hw, cy - hh),
        (cx + hw, cy - hh),
        (cx + hw, cy + hh),
        (cx - hw, cy + hh),
        (cx - hw, cy - hh),
    ]
}

fn rounded_rect_pts_ccw(cx: f32, cy: f32, w: f32, h: f32, r: f32) -> Vec<(f64, f64)> {
    let r_c = r.min(w * 0.5).min(h * 0.5).max(0.0);
    if r_c < 1e-4 {
        return rect_pts_ccw(cx, cy, w, h);
    }
    // Lots of segments per corner so the miter offset (used for the
    // beveled wall slopes) doesn't spike or kink at the arc joints.
    let segs = 32u32;
    let hw = (w * 0.5 - r_c) as f64;
    let hh = (h * 0.5 - r_c) as f64;
    let cx = cx as f64;
    let cy = cy as f64;
    let r = r_c as f64;

    let mut pts: Vec<(f64, f64)> = Vec::new();
    let mut corner = |center: (f64, f64), start: f64| {
        for i in 0..=segs {
            let t = i as f64 / segs as f64;
            let theta = start + t * std::f64::consts::FRAC_PI_2;
            pts.push((center.0 + theta.cos() * r, center.1 + theta.sin() * r));
        }
    };
    corner((cx + hw, cy - hh), -std::f64::consts::FRAC_PI_2);
    corner((cx + hw, cy + hh), 0.0);
    corner((cx - hw, cy + hh), std::f64::consts::FRAC_PI_2);
    corner((cx - hw, cy - hh), std::f64::consts::PI);
    let first = pts[0];
    pts.push(first);
    pts
}

fn circle_pts_ccw(cx: f32, cy: f32, r: f32, segments: u32) -> Vec<(f64, f64)> {
    let cx = cx as f64;
    let cy = cy as f64;
    let r = r as f64;
    let mut pts: Vec<(f64, f64)> = Vec::with_capacity(segments as usize + 1);
    for i in 0..segments {
        let theta = i as f64 / segments as f64 * std::f64::consts::TAU;
        pts.push((cx + theta.cos() * r, cy + theta.sin() * r));
    }
    pts.push(pts[0]);
    pts
}

fn polygon_filled(outer_ccw: Vec<(f64, f64)>) -> Polygon<f64> {
    Polygon::new(LineString::from(outer_ccw), vec![])
}

/// A frame: outer ring filled, with an inner ring as a hole. Both
/// rings should be supplied CCW; the inner one is reversed to CW
/// internally for geo's hole convention.
fn polygon_frame(outer_ccw: Vec<(f64, f64)>, mut inner_ccw: Vec<(f64, f64)>) -> Polygon<f64> {
    inner_ccw.reverse();
    Polygon::new(LineString::from(outer_ccw), vec![LineString::from(inner_ccw)])
}

fn rect_filled(cx: f32, cy: f32, w: f32, h: f32) -> Polygon<f64> {
    polygon_filled(rect_pts_ccw(cx, cy, w, h))
}

#[allow(dead_code)]
fn rect_frame(cx: f32, cy: f32, w: f32, h: f32, t: f32) -> Polygon<f64> {
    polygon_frame(
        rect_pts_ccw(cx, cy, w, h),
        rect_pts_ccw(cx, cy, (w - 2.0 * t).max(1e-4), (h - 2.0 * t).max(1e-4)),
    )
}

fn rounded_rect_frame(cx: f32, cy: f32, w: f32, h: f32, r: f32, t: f32) -> Polygon<f64> {
    polygon_frame(
        rounded_rect_pts_ccw(cx, cy, w, h, r),
        rounded_rect_pts_ccw(
            cx,
            cy,
            (w - 2.0 * t).max(1e-4),
            (h - 2.0 * t).max(1e-4),
            (r - t).max(0.0),
        ),
    )
}

fn circle_ring(cx: f32, cy: f32, outer_r: f32, t: f32, segments: u32) -> Polygon<f64> {
    let inner_r = (outer_r - t).max(1e-4);
    polygon_frame(
        circle_pts_ccw(cx, cy, outer_r, segments),
        circle_pts_ccw(cx, cy, inner_r, segments),
    )
}

fn triangle_filled(p1: Vec2, p2: Vec2, p3: Vec2) -> Polygon<f64> {
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

/// Carved form composed of fine linework — outlined frames, hairline
/// rules, ring-outlined radios and checkboxes. Stroke width and
/// element sizes are sized for a precise/blueprint look; the user is
/// expected to zoom in to see them clearly.
fn build_form_polygons() -> MultiPolygon<f64> {
    let mut polys: Vec<Polygon<f64>> = Vec::new();

    // Stroke widths.
    let line: f32 = 0.012;
    let frame: f32 = 0.014;

    // Outer frame.
    polys.push(rounded_rect_frame(0.0, 0.0, 4.6, 5.4, 0.10, frame));

    // Header rule below title area.
    polys.push(rect_filled(0.0, 2.10, 4.0, line));

    // Two text-input underlines (label area to the left would sit
    // here; we just carve the rule itself).
    polys.push(rect_filled(0.30, 1.50, 3.4, line));
    polys.push(rect_filled(0.30, 1.00, 3.4, line));

    // Three radio rings.
    for i in 0..3 {
        let x = -1.30 + i as f32 * 0.55;
        polys.push(circle_ring(x, 0.30, 0.090, line, 48));
    }

    // Three checkbox squares (outlined).
    for i in 0..3 {
        let x = -1.30 + i as f32 * 0.55;
        polys.push(rounded_rect_frame(x, -0.30, 0.180, 0.180, 0.018, line));
    }

    // Dropdown: underline rule + chevron glyph at the right end.
    polys.push(rect_filled(0.10, -0.95, 3.6, line));
    let chev_cx = 1.78;
    let chev_cy = -0.86;
    polys.push(triangle_filled(
        Vec2::new(chev_cx - 0.07, chev_cy),
        Vec2::new(chev_cx + 0.07, chev_cy),
        Vec2::new(chev_cx,        chev_cy - 0.085),
    ));

    // Button — small outlined rounded rect.
    polys.push(rounded_rect_frame(0.0, -1.65, 1.10, 0.36, 0.08, line));

    // Footer rule.
    polys.push(rect_filled(0.0, -2.20, 4.0, line));

    MultiPolygon(polys)
}

// ─── Icon polygon library ─────────────────────────────────────────
//
// Most icons (profile, mail, lock, eye, …) come from the SVG icon
// atlas built by `extract_svg_icons`. The arrow on the Sign Up
// button is still hand-coded because it's tightly coupled to the
// button's geometry and trivial to construct.

fn arrow_right_icon(cx: f32, cy: f32, size: f32) -> Vec<Polygon<f64>> {
    let head_w = size * 0.55;
    let head_h = size * 0.75;
    let tail_w = size * 0.55;
    let tail_h = size * 0.16;
    let head_tip_x = cx + size * 0.40;
    let head_base_x = head_tip_x - head_w;
    let tail_cx = head_base_x - tail_w * 0.5 + size * 0.05;

    let tail = rect_filled(tail_cx, cy, tail_w, tail_h);
    let head = triangle_filled(
        Vec2::new(head_tip_x, cy),
        Vec2::new(head_base_x, cy + head_h * 0.5),
        Vec2::new(head_base_x, cy - head_h * 0.5),
    );
    vec![tail, head]
}

// ─── Recreation of the "Create Account" reference image ──────────

/// Layout a single line of text as a list of glyph polygons, starting
/// at `origin` (baseline-left) at the given em size in world units.
fn layout_text(font_data: &[u8], text: &str, origin: Vec2, em: f32, tol: f32) -> Vec<Polygon<f64>> {
    let face = match Face::parse(font_data, 0) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let upem = face.units_per_em() as f32;
    let scale = em / upem;
    let mut polys = Vec::new();
    let mut x = origin.x;
    for ch in text.chars() {
        let Some(gid) = face.glyph_index(ch) else {
            x += em * 0.3;
            continue;
        };
        let advance = face.glyph_hor_advance(gid).unwrap_or(0) as f32 * scale;
        let path = outline_glyph_to_path(&face, gid, scale, Vec2::new(x, origin.y));
        let contours = flatten_path_to_contours(&path, tol);
        polys.extend(build_glyph_polygons(&contours));
        x += advance;
    }
    polys
}

fn pick_font<'a>(lib: &'a FontLibrary, names: &[&str]) -> Option<&'a NamedFont> {
    for name in names {
        if let Some(f) = lib.fonts.iter().find(|f| f.name == *name) {
            return Some(f);
        }
    }
    lib.fonts.first()
}

/// Build the "Create Account" reference as a RAISED stepped pyramid.
/// Each layer is a complete colored sheet sitting on top of the
/// previous one — not a hole carved into a bigger sheet. The base
/// (red) is largest; successive layers are smaller concentric
/// sheets stacking upward toward the centered purple card on top.
///
/// Layer plan (z increases upward; layer 0 is the bottom sheet):
///   0  red           — base, full paper.
///   1  orange        — first pyramid step.
///   2  yellow
///   3  green
///   4  teal
///   5  blue          — last rainbow ring.
///   6  dark purple   — card-shaped, peeks through form-field cutouts.
///   7  purple        — card top, with input/button/text/icon cutouts.
///
/// Walls between layers are vertical (no chamfer): the reference has
/// crisp edges between each color band, with ambient light dimming
/// the wall just enough that it reads as a clean line.
fn carve_create_account_form(stack: &mut PaperStack, font_lib: &FontLibrary, atlas: &IconAtlas) {
    stack.paper_thickness = 0.18;

    let n_layers = stack.layers.len();
    if n_layers < 9 {
        return;
    }

    // Palette (indexed 0..=8). The reference quantizes cleanly to
    // SIX rainbow bands (no separate green AND teal — what looks
    // green in the reference is a single teal-green band):
    //   0 = blue        (carve floor — innermost rainbow ring)
    //   1 = teal-green  (was: separate teal + green slots)
    //   2 = lime        (yellow-green)
    //   3 = yellow
    //   4 = orange
    //   5 = red         (outermost paper)
    //   6 = card dark purple   (text / icon recess)
    //   7 = card mid purple    (form-field bottom)
    //   8 = card light purple  (card top surface)
    //
    // Each rainbow color was inverse-solved from the actual rendered
    // output: starting from a target sampled from the reference, we
    // ran a screenshot, measured what came out, and offset the asked
    // sRGB by `(rendered - target)` so the next render lands on the
    // target. Every value here is darker than the reference RGB
    // because the lighting / ambient adds a ~+25-byte brightness on
    // top of whatever we ask for.
    let n_pal = stack.palette.len();
    if n_pal >= 9 {
        // Calibrated by /tmp/tune_loop.py over 12 iterations against
        // the cut-paper reference. Best mean Δ = 13.7 RGB-units.
        stack.palette[0] = Color::srgb(0.145, 0.231, 0.412); // blue        (target #2C4475)
        stack.palette[1] = Color::srgb(0.086, 0.545, 0.494); // teal-green  (target #308D7A)
        stack.palette[2] = Color::srgb(0.475, 0.706, 0.208); // lime        (target #7AA73F)
        stack.palette[3] = Color::srgb(1.000, 0.702, 0.000); // yellow      (target #E49F12)
        stack.palette[4] = Color::srgb(1.000, 0.416, 0.000); // orange      (target #F6701F)
        stack.palette[5] = Color::srgb(0.984, 0.208, 0.137); // red         (target #E54834)
        // Card purples filled in below.
        for i in 9..n_pal {
            stack.palette[i] = stack.palette[5];
        }
    }

    // Reset all layers and slopes (vertical walls = crisp edges).
    for layer in stack.layers.iter_mut() {
        *layer = MultiPolygon(vec![]);
    }
    for s in stack.layer_slope_inset.iter_mut() {
        *s = 0.0;
    }

    // Explicit z heights. Carved-down rainbow: red highest, blue
    // lowest. Card sits ON blue and rises back UP to roughly the
    // teal-green / lime / yellow heights (in different XY).
    let pt = stack.paper_thickness;
    let z_blue   = 1.0 * pt; // floor
    let z_teal   = 2.0 * pt;
    let z_lime   = 3.0 * pt;
    let z_yellow = 4.0 * pt;
    let z_orange = 5.0 * pt;
    let z_red    = 6.0 * pt; // top
    stack.layer_z[0] = z_blue;
    stack.layer_z[1] = z_teal;
    stack.layer_z[2] = z_lime;
    stack.layer_z[3] = z_yellow;
    stack.layer_z[4] = z_orange;
    stack.layer_z[5] = z_red;

    // Geometry sizes.
    let paper_w: f32 = 18.0;
    let paper = polygon_filled(rounded_rect_pts_ccw(0.0, 0.0, paper_w, paper_w, 0.5));

    // Each carve is the SAME shape size at successive higher z. Going
    // from inside-out (smallest carve to biggest), each step grows by
    // `step` so the rings exposed between carves are uniform width.
    let smallest_carve_w: f32 = 7.4;
    let step: f32 = 0.55;
    let carve = |size: f32, radius: f32| -> Polygon<f64> {
        polygon_filled(rounded_rect_pts_ccw(0.0, 0.0, size, size, radius))
    };
    let carve_teal   = carve(smallest_carve_w + 0.0  * step, 0.42);
    let carve_lime   = carve(smallest_carve_w + 1.0  * step, 0.46);
    let carve_yellow = carve(smallest_carve_w + 2.0  * step, 0.50);
    let carve_orange = carve(smallest_carve_w + 3.0  * step, 0.54);
    let carve_red    = carve(smallest_carve_w + 4.0  * step, 0.58);

    let mp = |p: Polygon<f64>| MultiPolygon(vec![p]);

    // Layer 0: blue = full paper, no carve. The floor that the card
    // sits on.
    stack.layers[0] = MultiPolygon(vec![paper.clone()]);

    // Layers 1..=5: each is full paper MINUS its respective carve.
    stack.layers[1] = mp(paper.clone()).difference(&mp(carve_teal.clone()));
    stack.layers[2] = mp(paper.clone()).difference(&mp(carve_lime));
    stack.layers[3] = mp(paper.clone()).difference(&mp(carve_yellow));
    stack.layers[4] = mp(paper.clone()).difference(&mp(carve_orange));
    stack.layers[5] = mp(paper.clone()).difference(&mp(carve_red));

    // Card body. Smaller than the smallest rainbow carve so blue
    // floor is visible as a ring around the card.
    let card_w: f32 = 5.6;
    let card_h: f32 = 5.6;
    let card_r: f32 = 0.30;
    let card_shape = polygon_filled(rounded_rect_pts_ccw(0.0, 0.0, card_w, card_h, card_r));

    // Layout (relative to card center).
    let avatar_x: f32 = -2.0;
    let avatar_y: f32 = 1.95;
    let avatar_r: f32 = 0.32;

    let input_w: f32 = 4.5;
    let input_h: f32 = 0.50;
    let input_r: f32 = 0.13;
    let input_ys: [f32; 3] = [1.05, 0.40, -0.25];

    let row_y: f32 = -0.95;
    let checkbox_size: f32 = 0.26;

    let button_w: f32 = 4.5;
    let button_h: f32 = 0.62;
    let button_r: f32 = 0.16;
    let button_y: f32 = -1.75;

    // Per-target-layer cut buckets. Each feature stamps into
    // exactly *one* layer (the layer immediately below the surface
    // it sits on); cumulative recess depth comes from features
    // overlapping in XY, not from one feature cutting many layers.
    //
    //   cuts[0] → cut layer 8 (visible bottom = layer 7, mid purple)
    //   cuts[1] → cut layer 7 (visible bottom = layer 6, deep purple)
    //   cuts[2] → cut layer 6 (visible bottom = blue floor)
    //
    // Anything deeper clamps to cuts[2].
    let mut cuts: [Vec<Polygon<f64>>; 3] = Default::default();

    // Form fields, checkbox, button — 1-cut features on the bare
    // card → bucket 0 (cut layer 8).
    for &y in &input_ys {
        cuts[0].push(polygon_filled(rounded_rect_pts_ccw(
            0.0, y, input_w, input_h, input_r,
        )));
    }
    cuts[0].push(polygon_filled(rounded_rect_pts_ccw(
        -2.00, row_y, checkbox_size, checkbox_size, 0.04,
    )));
    cuts[0].push(polygon_filled(rounded_rect_pts_ccw(
        0.0, button_y, button_w, button_h, button_r,
    )));

    // Icons. Each icon is one feature, just like text: it cuts ONE
    // layer — the layer it sits on. The atlas's per-pixel depth
    // values are used only as a way to define the icon's overall
    // shape (union of every colored pixel, regardless of which
    // legend swatch it matched). If we want multi-level relief
    // *inside* an icon later, we can revisit; right now an icon
    // recesses by exactly one step like every other element.
    let push_atlas = |name: &str,
                      pos: Vec2,
                      size_h: f32,
                      base_depth: usize,
                      cuts: &mut [Vec<Polygon<f64>>; 3]| {
        let layers = icon_at(atlas, name, pos, size_h);
        let mut all_polys: Vec<Polygon<f64>> = Vec::new();
        for (_depth, polys) in layers {
            all_polys.extend(polys);
        }
        cuts[base_depth.min(2)].extend(all_polys);
    };
    let avatar_size_h = avatar_r * 2.0;
    // Avatar sits on the card surface — base_depth 0.
    push_atlas(
        "profile_circle",
        Vec2::new(avatar_x, avatar_y),
        avatar_size_h,
        0,
        &mut cuts,
    );
    let icon_size_h: f32 = 0.42;
    let icon_x: f32 = -1.86;
    // Field-row icons sit inside their respective form-field
    // cutouts, which already cut 1 layer → base_depth 1.
    push_atlas(
        "profile_simple",
        Vec2::new(icon_x, input_ys[0]),
        icon_size_h,
        1,
        &mut cuts,
    );
    push_atlas(
        "mail",
        Vec2::new(icon_x, input_ys[1]),
        icon_size_h,
        1,
        &mut cuts,
    );
    push_atlas(
        "lock",
        Vec2::new(icon_x, input_ys[2]),
        icon_size_h,
        1,
        &mut cuts,
    );
    push_atlas(
        "eye",
        Vec2::new(2.00, input_ys[2]),
        icon_size_h,
        1,
        &mut cuts,
    );
    // Hand-drawn arrow_right inside the Sign Up button — 1-cut
    // feature at base_depth 1 → bucket 1.
    cuts[1].extend(arrow_right_icon(1.80, button_y, 0.30));

    // Text. Each label is a 1-cut feature: cuts ONE layer below the
    // surface it sits on. Texts on the bare card go to bucket 0
    // (cut layer 8 → visible at mid). Texts inside form fields /
    // the button go to bucket 1 (cut layer 7 → visible at deep,
    // since the field/button has already cut layer 8).
    let mut push_text = |text: &str,
                         pos: Vec2,
                         em: f32,
                         tol: f64,
                         base_depth: usize,
                         cuts: &mut [Vec<Polygon<f64>>; 3]| {
        if let Some(font) = pick_font(font_lib, &["Arial Bold", "Helvetica", "Arial"]) {
            let polys = layout_text(&font.data, text, pos, em, tol as f32);
            cuts[base_depth.min(2)].extend(polys);
        }
    };
    let tol = 0.005;
    // On-card titles + side text → base 0.
    push_text("Create Account", Vec2::new(-1.30, 2.10), 0.40, tol, 0, &mut cuts);
    push_text("Let's get you started.", Vec2::new(-1.30, 1.75), 0.21, tol, 0, &mut cuts);
    push_text("Remember me", Vec2::new(-1.75, row_y - 0.08), 0.20, tol, 0, &mut cuts);
    push_text("Forgot password?", Vec2::new(0.50, row_y - 0.08), 0.20, tol, 0, &mut cuts);
    // Field labels sit inside the form-field rectangles → base 1.
    push_text("Full Name", Vec2::new(-1.55, input_ys[0] - 0.08), 0.26, tol, 1, &mut cuts);
    push_text("Email Address", Vec2::new(-1.55, input_ys[1] - 0.08), 0.26, tol, 1, &mut cuts);
    push_text("Password", Vec2::new(-1.55, input_ys[2] - 0.08), 0.26, tol, 1, &mut cuts);
    // Sign Up sits inside the button → base 1.
    push_text("Sign Up", Vec2::new(-0.65, button_y - 0.11), 0.28, tol, 1, &mut cuts);

    // THREE purple layers stacked on the blue floor:
    //   layer 6 = deep purple (z=z_teal)
    //   layer 7 = mid purple  (z=z_lime)
    //   layer 8 = top purple  (z=z_yellow) — bare card surface
    //
    // Each layer is the card shape MINUS the features targeting
    // that layer (built up in `cuts[0..2]` above). Cumulative
    // visibility comes from features that overlap in XY: e.g. a
    // text glyph inside a form field cuts only layer 7, but the
    // form field has already cut layer 8 there, so the bottom of
    // the well at the glyph position is layer 6 (deep purple).
    if n_layers < 9 {
        return;
    }
    if n_pal >= 9 {
        // Card purples — calibrated against reference samples
        // (top #8761AB, mid #69408D, deep #35214B).
        // #35214B
        stack.palette[6] = Color::srgb(0.208, 0.129, 0.294); // deep
        // #69408D
        stack.palette[7] = Color::srgb(0.412, 0.251, 0.553); // mid
        // #8761AB
        stack.palette[8] = Color::srgb(0.529, 0.380, 0.671); // top
    }
    stack.layer_z[6] = z_teal;
    stack.layer_z[7] = z_lime;
    stack.layer_z[8] = z_yellow;

    // Build subtractions iteratively. Geo's difference with a
    // MultiPolygon containing OVERLAPPING subtrahends (e.g. button
    // rect that already contains the Sign Up text) can drop subtle
    // holes; subtracting one polygon at a time is more reliable.
    let subtract_each = |start: MultiPolygon<f64>,
                         shapes: &[Polygon<f64>]|
     -> MultiPolygon<f64> {
        let mut acc = start;
        for s in shapes {
            acc = acc.difference(&MultiPolygon(vec![s.clone()]));
        }
        acc
    };

    // Each card layer subtracts only the features that target it.
    stack.layers[8] = subtract_each(mp(card_shape.clone()), &cuts[0]);
    stack.layers[7] = subtract_each(mp(card_shape.clone()), &cuts[1]);
    stack.layers[6] = subtract_each(mp(card_shape), &cuts[2]);

    // Empty + push out-of-the-way any layers above 8.
    for i in 9..n_layers {
        stack.layers[i] = MultiPolygon(vec![]);
        stack.layer_z[i] = z_red + (i as f32 - 5.0) * pt;
    }
}

// ─── Auto-screenshot ──────────────────────────────────────────────

#[derive(Resource)]
struct AutoScreenshot {
    path: String,
    state: AutoScreenshotState,
}

enum AutoScreenshotState {
    /// Frames to wait so the mesh has time to regenerate and render.
    Waiting(u32),
    /// Screenshot has been spawned; wait this many frames for the
    /// async write to finish before exiting.
    AfterCapture(u32),
    Done,
}

fn auto_screenshot_system(
    mut commands: Commands,
    auto: Option<ResMut<AutoScreenshot>>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(mut auto) = auto else { return; };
    match &mut auto.state {
        AutoScreenshotState::Waiting(n) => {
            if *n > 0 {
                *n -= 1;
                return;
            }
            let path = auto.path.clone();
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk(path));
            auto.state = AutoScreenshotState::AfterCapture(60);
        }
        AutoScreenshotState::AfterCapture(n) => {
            if *n > 0 {
                *n -= 1;
                return;
            }
            exit.write(AppExit::Success);
            auto.state = AutoScreenshotState::Done;
            // Belt-and-braces: AppExit alone sometimes leaves the
            // window lingering on macOS — force-exit the process so
            // the window goes down immediately.
            std::process::exit(0);
        }
        AutoScreenshotState::Done => {}
    }
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
    debug_view: Res<DebugView>,
    windows: Query<&Window>,
) {
    let dt = time.delta_secs();

    // egui swallows trackpad input over its panels.
    let egui_has_pointer = contexts
        .ctx_mut()
        .map(|c| c.is_pointer_over_area() || c.wants_pointer_input())
        .unwrap_or(false);

    // In split-view mode the right half of the window belongs to the
    // scene-view camera; pan/zoom input there must not move the book.
    let cursor_in_scene = debug_view.enabled
        && windows
            .single()
            .ok()
            .and_then(|w| w.cursor_position().map(|c| c.x >= w.width() * 0.5))
            .unwrap_or(false);
    let block_pointer_input = egui_has_pointer || cursor_in_scene;

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
        if block_pointer_input { continue; }
        let factor = match event.unit {
            MouseScrollUnit::Pixel => 0.005,
            MouseScrollUnit::Line  => 0.10,
        };
        // Vertical follows the user's reported feel; horizontal is
        // inverted from the wheel-event sign so the paper tracks the
        // finger direction.
        pan_delta.x -= event.x * factor;
        pan_delta.y += event.y * factor;
    }
    for event in pan_gesture.read() {
        if block_pointer_input { continue; }
        pan_delta.x -= event.0.x * 0.005;
        pan_delta.y += event.0.y * 0.005;
    }
    if pan_delta != Vec2::ZERO {
        let z = camera.zoom;
        camera.pan += pan_delta / z;
    }

    // Pinch: positive delta = pinch out = zoom in.
    for event in pinch.read() {
        if block_pointer_input { continue; }
        camera.zoom = (camera.zoom * (1.0 + event.0)).clamp(0.1, 64.0);
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
    cameras: Query<(&Camera, &GlobalTransform), (With<Camera3d>, With<MainCam>)>,
    stack: Res<PaperStack>,
    mut hover: ResMut<HoverPos>,
    mut contexts: EguiContexts,
    debug_view: Res<DebugView>,
) {
    if let Ok(ctx) = contexts.ctx_mut() {
        if ctx.is_pointer_over_area() {
            hover.0 = None;
            return;
        }
    }
    let Ok(window) = windows.single() else { hover.0 = None; return; };
    let Some(cursor) = window.cursor_position() else { hover.0 = None; return; };
    // In split-view mode the cursor only addresses the book UI when
    // it's over the left half of the window.
    if debug_view.enabled && cursor.x > window.width() * 0.5 {
        hover.0 = None;
        return;
    }
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
        dl.illuminance = l.intensity * 7_500.0;
        dl.shadows_enabled = on && l.casts_shadow;
    }
}

fn sync_ambient(light: Res<LightControl>, mut ambient: ResMut<GlobalAmbientLight>) {
    ambient.brightness = light.ambient * 1500.0;
}

fn sync_camera(
    camera_ctrl: Res<CameraControl>,
    debug_view: Res<DebugView>,
    mut q: Query<&mut Transform, (With<Camera3d>, With<MainCam>)>,
) {
    let Ok(mut tf) = q.single_mut() else { return };
    let target = Vec3::new(camera_ctrl.pan.x, camera_ctrl.pan.y, 0.0);
    let base_distance = 14.0;
    let distance = base_distance / camera_ctrl.zoom;
    // When the debug 3D view is active the left viewport is locked to
    // top-down so the user has a stable plan-view of the book UI.
    let tilt = if debug_view.enabled { 0.0 } else { camera_ctrl.tilt };
    let offset = Quat::from_rotation_x(-tilt) * Vec3::new(0.0, 0.0, distance);
    *tf = Transform::from_translation(target + offset).looking_at(target, Vec3::Y);
}

/// Compact 2D direction pad for a directional light. The pad is a
/// top-down projection of the upper hemisphere: center = overhead,
/// edge = grazing, +y on screen = +y in world. Click/drag to set the
/// direction; Z is auto-derived as the remaining elevation. Returns
/// `true` if the user changed the value.
fn light_dir_pad(ui: &mut egui::Ui, dir: &mut Vec3, dot_color: egui::Color32) -> bool {
    let size = egui::vec2(72.0, 72.0);
    let (resp, painter) = ui.allocate_painter(size, egui::Sense::click_and_drag());
    let rect = resp.rect;
    let center = rect.center();
    let r = rect.width().min(rect.height()) * 0.5 - 3.0;

    let bg = egui::Color32::from_rgb(28, 28, 32);
    let grid = egui::Color32::from_rgb(70, 70, 80);
    painter.circle_filled(center, r, bg);
    painter.circle_stroke(center, r, egui::Stroke::new(1.0, grid));
    painter.circle_stroke(center, r * 0.5, egui::Stroke::new(0.5, grid));
    painter.line_segment(
        [egui::pos2(center.x - r, center.y), egui::pos2(center.x + r, center.y)],
        egui::Stroke::new(0.5, grid),
    );
    painter.line_segment(
        [egui::pos2(center.x, center.y - r), egui::pos2(center.x, center.y + r)],
        egui::Stroke::new(0.5, grid),
    );

    // Stored l.dir is in shader-space (y-down). Project to world for
    // the pad: world = (dir.x, -dir.y, dir.z), normalized.
    let world = Vec3::new(dir.x, -dir.y, dir.z)
        .try_normalize()
        .unwrap_or(Vec3::Z);
    let dot_pos = egui::pos2(center.x + world.x * r, center.y - world.y * r);
    painter.circle_filled(dot_pos, 5.0, dot_color);
    painter.circle_stroke(dot_pos, 5.0, egui::Stroke::new(1.0, egui::Color32::WHITE));

    let mut changed = false;
    if (resp.clicked() || resp.dragged()) && let Some(pos) = resp.interact_pointer_pos() {
        let mut u = (pos.x - center.x) / r;
        let mut v = -(pos.y - center.y) / r;
        let len = (u * u + v * v).sqrt();
        if len > 1.0 {
            u /= len;
            v /= len;
        }
        let z = (1.0 - (u * u + v * v)).max(0.0).sqrt();
        *dir = Vec3::new(u, -v, z);
        changed = true;
    }
    changed
}

// ─── Debug 3D scene view systems ──────────────────────────────────

/// Convert the stored "shader-space, y-down" light direction into a
/// world-space unit vector pointing FROM the origin TOWARD the light.
fn light_dir_to_world(dir: Vec3) -> Vec3 {
    Vec3::new(dir.x, -dir.y, dir.z).try_normalize().unwrap_or(Vec3::Z)
}

/// Inverse of [`light_dir_to_world`].
fn world_to_light_dir(world: Vec3) -> Vec3 {
    let n = world.try_normalize().unwrap_or(Vec3::Z);
    Vec3::new(n.x, -n.y, n.z)
}

/// Resize each camera's viewport to either fullscreen (debug view
/// off) or split halves (debug view on). Run every frame so window
/// resizes flow through.
fn apply_viewport_layout(
    debug_view: Res<DebugView>,
    windows: Query<&Window>,
    mut main_q: Query<&mut Camera, (With<MainCam>, Without<SceneCam>)>,
    mut scene_q: Query<&mut Camera, (With<SceneCam>, Without<MainCam>)>,
) {
    let Ok(window) = windows.single() else { return };
    let w = window.physical_width().max(2);
    let h = window.physical_height().max(2);

    let Ok(mut main_cam) = main_q.single_mut() else { return };
    let Ok(mut scene_cam) = scene_q.single_mut() else { return };

    if debug_view.enabled {
        let half = w / 2;
        main_cam.viewport = Some(Viewport {
            physical_position: UVec2::ZERO,
            physical_size: UVec2::new(half, h),
            ..default()
        });
        scene_cam.viewport = Some(Viewport {
            physical_position: UVec2::new(half, 0),
            physical_size: UVec2::new(w - half, h),
            ..default()
        });
        scene_cam.is_active = true;
    } else {
        main_cam.viewport = None;
        scene_cam.viewport = None;
        scene_cam.is_active = false;
    }
}

/// Position the scene-view camera from orbit params (yaw/pitch/dist
/// around origin), Z-up.
fn sync_scene_cam(
    debug_view: Res<DebugView>,
    mut q: Query<&mut Transform, (With<SceneCam>, Without<MainCam>)>,
) {
    if !debug_view.enabled {
        return;
    }
    let Ok(mut tf) = q.single_mut() else { return };
    let cp = debug_view.pitch.cos();
    let offset = Vec3::new(
        debug_view.dist * cp * debug_view.yaw.sin(),
        debug_view.dist * cp * debug_view.yaw.cos(),
        debug_view.dist * debug_view.pitch.sin(),
    );
    *tf = Transform::from_translation(offset).looking_at(Vec3::ZERO, Vec3::Z);
}

/// Position/scale/colorize each light gizmo from `LightControl`.
fn sync_light_gizmos(
    light: Res<LightControl>,
    debug_view: Res<DebugView>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut q: Query<(
        &LightGizmo,
        &LightGizmoMaterial,
        &mut Transform,
        &mut Visibility,
    )>,
) {
    for (idx, mat_h, mut tf, mut vis) in q.iter_mut() {
        *vis = if debug_view.enabled {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        let l = light.lights[idx.0];
        let world_dir = light_dir_to_world(l.dir);
        let pos = world_dir * LIGHT_GIZMO_DOME_R;
        let selected = debug_view.selected_light == Some(idx.0);
        let scale = if selected { 1.6 } else { 1.0 } * (0.6 + l.intensity.min(2.0) * 0.5);
        *tf = Transform::from_translation(pos).with_scale(Vec3::splat(scale));

        if let Some(m) = materials.get_mut(&mat_h.0) {
            let c = l.color().to_linear();
            // Boost emission with intensity so brighter lights glow more.
            let k = 0.6 + l.intensity.min(2.0) * 1.4;
            m.base_color = Color::linear_rgb(c.red, c.green, c.blue);
            m.emissive = LinearRgba::new(c.red * k, c.green * k, c.blue * k, 1.0);
        }
    }
}

/// Ray–sphere intersection (returns the closer positive t if any).
fn ray_sphere_t(origin: Vec3, dir: Vec3, center: Vec3, radius: f32) -> Option<f32> {
    let oc = origin - center;
    let b = oc.dot(dir);
    let c = oc.length_squared() - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let s = disc.sqrt();
    let t1 = -b - s;
    let t2 = -b + s;
    if t1 > 1e-4 {
        Some(t1)
    } else if t2 > 1e-4 {
        Some(t2)
    } else {
        None
    }
}

/// Cursor in the scene viewport drives:
///   • left-click on a gizmo → drag along the dome (updates light dir)
///   • right-drag → orbit yaw/pitch
///   • wheel → dolly in/out
fn scene_view_input(
    mut debug_view: ResMut<DebugView>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut motion: MessageReader<MouseMotion>,
    mut wheel: MessageReader<MouseWheel>,
    windows: Query<&Window>,
    cameras: Query<(&Camera, &GlobalTransform), (With<SceneCam>, Without<MainCam>)>,
    mut light: ResMut<LightControl>,
    mut contexts: EguiContexts,
) {
    if !debug_view.enabled {
        motion.clear();
        return;
    }
    let egui_has_pointer = contexts
        .ctx_mut()
        .map(|c| c.is_pointer_over_area() || c.wants_pointer_input())
        .unwrap_or(false);

    let Ok(window) = windows.single() else { return };
    let Ok((camera, cam_tf)) = cameras.single() else { return };
    let cursor = window.cursor_position();
    let cursor_in_scene = cursor
        .map(|c| c.x >= window.width() * 0.5)
        .unwrap_or(false);

    // ── Left click: pick a gizmo and start dragging ─────────────────
    if !debug_view.dragging_light
        && mouse.just_pressed(MouseButton::Left)
        && cursor_in_scene
        && !egui_has_pointer
    {
        if let Some(c) = cursor {
            if let Ok(ray) = camera.viewport_to_world(cam_tf, c) {
                let origin = ray.origin;
                let dir = ray.direction.as_vec3();
                let mut best: Option<(usize, f32)> = None;
                for (i, l) in light.lights.iter().enumerate() {
                    let center = light_dir_to_world(l.dir) * LIGHT_GIZMO_DOME_R;
                    if let Some(t) = ray_sphere_t(origin, dir, center, LIGHT_GIZMO_PICK_R) {
                        if best.map_or(true, |(_, bt)| t < bt) {
                            best = Some((i, t));
                        }
                    }
                }
                if let Some((i, _)) = best {
                    debug_view.selected_light = Some(i);
                    debug_view.dragging_light = true;
                }
            }
        }
    }

    // ── Drag: project ray onto the dome, update light direction ─────
    if debug_view.dragging_light && mouse.pressed(MouseButton::Left) {
        if let (Some(c), Some(idx)) = (cursor, debug_view.selected_light) {
            if let Ok(ray) = camera.viewport_to_world(cam_tf, c) {
                let origin = ray.origin;
                let dir = ray.direction.as_vec3();
                let new_pos = ray_sphere_t(origin, dir, Vec3::ZERO, LIGHT_GIZMO_DOME_R)
                    .map(|t| origin + dir * t)
                    .unwrap_or_else(|| {
                        // Ray misses the dome — use closest approach to origin.
                        let t = (-origin).dot(dir);
                        let p = origin + dir * t.max(0.1);
                        p.try_normalize().unwrap_or(Vec3::Z) * LIGHT_GIZMO_DOME_R
                    });
                light.lights[idx].dir = world_to_light_dir(new_pos);
            }
        }
    }

    if mouse.just_released(MouseButton::Left) {
        debug_view.dragging_light = false;
    }

    // ── Right-drag: orbit yaw/pitch ─────────────────────────────────
    if mouse.just_pressed(MouseButton::Right) && cursor_in_scene && !egui_has_pointer {
        debug_view.orbiting = true;
    }
    if mouse.just_released(MouseButton::Right) {
        debug_view.orbiting = false;
    }
    if debug_view.orbiting {
        let mut delta = Vec2::ZERO;
        for ev in motion.read() {
            delta += ev.delta;
        }
        debug_view.yaw -= delta.x * 0.005;
        debug_view.pitch = (debug_view.pitch + delta.y * 0.005)
            .clamp(-1.45, 1.45);
    } else {
        motion.clear();
    }

    // ── Wheel in scene viewport: dolly ──────────────────────────────
    if cursor_in_scene && !egui_has_pointer {
        for ev in wheel.read() {
            let factor = match ev.unit {
                MouseScrollUnit::Pixel => 0.01,
                MouseScrollUnit::Line => 0.2,
            };
            debug_view.dist = (debug_view.dist * (1.0 - ev.y * factor)).clamp(4.0, 80.0);
        }
    }
}

/// Toggle the split-view debug interface with `G`.
fn toggle_debug_view(
    keys: Res<ButtonInput<KeyCode>>,
    mut debug_view: ResMut<DebugView>,
) {
    if keys.just_pressed(KeyCode::KeyG) {
        debug_view.enabled = !debug_view.enabled;
    }
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
    icon_atlas: Res<IconAtlas>,
    mut debug_view: ResMut<DebugView>,
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
                ui.add(
                    egui::Slider::new(&mut brush.radius, 0.002..=1.0)
                        .logarithmic(true)
                        .text("radius"),
                );
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

            ui.collapsing("Demo scenes", |ui| {
                if ui.button("Show \"Create Account\" demo").clicked() {
                    stack.reset_full();
                    carve_create_account_form(&mut stack, &font_lib, &icon_atlas);
                    // Frame it: top-down, zoomed out enough to fit the
                    // 9.6-wide outer carve plus a margin of red paper.
                    camera.pan = Vec2::ZERO;
                    camera.zoom = 0.58;
                    camera.tilt = 0.0;
                    dirty.0 = true;
                }
                ui.label(
                    "Carves the full layered reference scene and\n\
                     sets the camera to frame it top-down.",
                );
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
                if ui
                    .add(
                        egui::Slider::new(&mut stack.chamfer_w_frac, 0.0..=1.0)
                            .text("chamfer width (× thickness)"),
                    )
                    .changed()
                {
                    dirty.0 = true;
                }
                if ui
                    .add(
                        egui::Slider::new(&mut stack.chamfer_h_frac, 0.0..=1.0)
                            .text("chamfer height (× thickness)"),
                    )
                    .changed()
                {
                    dirty.0 = true;
                }
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
                            ui.checkbox(&mut l.casts_shadow, "casts shadow");
                            ui.add(egui::Slider::new(&mut l.intensity, 0.0..=2.0).text("intensity"));
                            let lc = l.color().to_linear();
                            let dot_color = egui::Color32::from_rgb(
                                (lc.red.clamp(0.0, 1.0) * 255.0) as u8,
                                (lc.green.clamp(0.0, 1.0) * 255.0) as u8,
                                (lc.blue.clamp(0.0, 1.0) * 255.0) as u8,
                            );
                            ui.horizontal(|ui| {
                                light_dir_pad(ui, &mut l.dir, dot_color);
                                ui.vertical(|ui| {
                                    ui.add(
                                        egui::DragValue::new(&mut l.dir.x)
                                            .speed(0.01)
                                            .range(-1.0..=1.0)
                                            .prefix("x "),
                                    );
                                    ui.add(
                                        egui::DragValue::new(&mut l.dir.y)
                                            .speed(0.01)
                                            .range(-1.0..=1.0)
                                            .prefix("y "),
                                    );
                                    ui.add(
                                        egui::DragValue::new(&mut l.dir.z)
                                            .speed(0.01)
                                            .range(0.0..=1.0)
                                            .prefix("z "),
                                    );
                                });
                            });
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
                ui.add(
                    egui::Slider::new(&mut camera.zoom, 0.1..=64.0)
                        .logarithmic(true)
                        .text("zoom"),
                );
                let mut tilt_deg = camera.tilt.to_degrees();
                if ui
                    .add(egui::Slider::new(&mut tilt_deg, 0.0..=80.0).text("tilt (°)"))
                    .changed()
                {
                    camera.tilt = tilt_deg.to_radians();
                }
            });

            ui.collapsing("Debug 3D scene view", |ui| {
                ui.checkbox(&mut debug_view.enabled, "split-view (G)");
                ui.label(
                    "Left = book UI top-down · right = 3D scene with\n\
                     light gizmos. Drag a sphere to move that light;\n\
                     right-drag to orbit; wheel to dolly.",
                );
                let mut yaw_deg = debug_view.yaw.to_degrees();
                if ui
                    .add(egui::Slider::new(&mut yaw_deg, -180.0..=180.0).text("orbit yaw (°)"))
                    .changed()
                {
                    debug_view.yaw = yaw_deg.to_radians();
                }
                let mut pitch_deg = debug_view.pitch.to_degrees();
                if ui
                    .add(egui::Slider::new(&mut pitch_deg, -85.0..=85.0).text("orbit pitch (°)"))
                    .changed()
                {
                    debug_view.pitch = pitch_deg.to_radians();
                }
                ui.add(egui::Slider::new(&mut debug_view.dist, 4.0..=80.0).text("orbit dist"));
                if let Some(i) = debug_view.selected_light {
                    ui.label(format!("selected: light {}", i + 1));
                } else {
                    ui.label("selected: (none)");
                }
            });

            ui.separator();
            ui.label("Tab to hide. Two-finger pan, pinch zoom, [ / ] tilt. G toggles 3D view.");
        });
    Ok(())
}
