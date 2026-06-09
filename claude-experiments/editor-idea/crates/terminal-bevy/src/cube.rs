//! The project prism — a Compiz-desktop-cube-style 3D overview where
//! each pane floats in 3D off its project's face.
//!
//! ## What it does
//!
//! Press the toggle (Cmd+Shift+\) and the flat canvas animates OUT into a
//! rotating N-sided prism. Each project is one vertical face (a dim
//! "surface" backboard) and that project's panes float OUT from it at
//! staggered depths, each at natural size (independent of the project's
//! pan/zoom). The pane textures are *live*. Drag to rotate, scroll to
//! dolly. Click a project to fly into it (becomes the active project).
//! Escape flies back into the project you came from.
//!
//! ## How it works (the Compiz trick, per-pane)
//!
//! One live render-target texture per *pane*: a DEDICATED `Camera2d` per
//! pane (separate from the pane's own window camera) renders that pane's
//! render layer into its own image, aimed at the pane's ACTUAL flat
//! position (derived from the live `PaneViewport`). The panes themselves
//! are never moved or hidden — they stay exactly where the flat editor
//! has them, harmlessly behind the cube — so opening/closing the overview
//! involves no pane reposition and therefore no one-frame "jump". Each
//! pane becomes a 3D quad on its project's face, pushed out along the face
//! normal. A `Camera3d` (order 1,000,000) draws the prism over everything;
//! keyboard focus is parked so nothing leaks into terminals while up.

use std::collections::BTreeMap;
use std::collections::HashMap;

use bevy::asset::{embedded_path, AssetPath};
use bevy::camera::visibility::RenderLayers;
use bevy::camera::{Camera, ClearColorConfig, RenderTarget};
use bevy::image::Image;
use bevy::input::keyboard::KeyboardInput;
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::pbr::{Material, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, TextureFormat};
use bevy::shader::ShaderRef;

use pane_bevy::{
    FocusedPane, InputConsumed, PaneCameraOf, PaneLayer, PanePinned, PaneProject, PaneRect, PaneTag,
};

use crate::projects::Projects;

// ---------- Constants ----------

/// Render layer for the prism's structural geometry (boards, face quads,
/// reflections) — drawn globally by the cube's 3D camera. Reserved in
/// `PaneLayerAllocator` (see `PanePlugin` setup) so a pane is NEVER
/// allocated it; otherwise that pane's content would be drawn by the cube
/// camera across the whole overview.
pub(crate) const CUBE_LAYER: usize = 4096;

fn render_layer(n: usize) -> RenderLayers {
    RenderLayers::from_layers(&[n])
}

/// Above EVERY other window camera (incl. the menu overlay at 100_000).
const CUBE_CAM_ORDER: isize = 1_000_000;

const FACE_HEIGHT: f32 = 4.0;
/// Backboard sits slightly BEHIND the face plane so no pane (which all
/// float at >= 0) ever z-fights it.
const BOARD_RECESS: f32 = 0.08;
/// Per-z-order depth separation. Kept TINY: at the dive endpoint the
/// camera is close and in perspective, so a larger offset makes high-z
/// panes render visibly larger/shifted than their flat position — they'd
/// snap (jump) when the flat view takes over. The tight near/far gives
/// enough depth precision that this small value still avoids z-fighting.
const DEPTH_EPS: f32 = 0.004;
/// Small lift so pinned panes sit just in front of the recessed board.
const PIN_LIFT: f32 = 0.01;

/// Vertical FOV of the cube camera (degrees). The fill distance (where a
/// face exactly fills the window) is derived from it.
const FOV_DEG: f32 = 45.0;

/// Longest-side pixel cap for per-pane render-target textures. Panes are
/// small in the overview, so we don't need full native resolution — this
/// is the dominant lever on the overview's memory footprint.
const MAX_FACE_TEX: f32 = 1280.0;

const BACKDROP: Color = Color::srgb(0.04, 0.05, 0.07);

// ---------- Procedural sky shader ----------

/// Material for the procedural skybox. `params.x` = mode, `params.y` =
/// time. Driven by `sky.wgsl` (embedded, runtime GPU shader).
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct SkyMaterial {
    #[uniform(0)]
    pub params: Vec4,
}

impl Material for SkyMaterial {
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Path(AssetPath::from_path_buf(embedded_path!("sky.wgsl")).with_source("embedded"))
    }
}

fn skybox_mode(name: &str) -> f32 {
    match name {
        "nebula" => 1.0,
        "space" => 2.0,
        "aurora" => 3.0,
        _ => 0.0, // dusk
    }
}

/// Reflective shiny-floor material. `params.xyz` = floor color, `params.w`
/// = reflection strength. See `floor.wgsl`.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct FloorMaterial {
    #[uniform(0)]
    pub params: Vec4,
}

impl Material for FloorMaterial {
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Path(AssetPath::from_path_buf(embedded_path!("floor.wgsl")).with_source("embedded"))
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }
}


// ---------- Runtime-tunable params (hot-reloaded from Rhai) ----------

/// Live-tunable overview parameters, driven by `~/.jim/cube.rhai`.
/// `board_recede` and `cam_tilt`/durations read every frame (apply
/// instantly); float/fill/pullback bake in on the next overview open.
#[derive(Resource, Clone, Debug)]
pub struct CubeParams {
    pub board_recede: f32,
    pub base_float: f32,
    pub float_step: f32,
    pub overview_pullback: f32,
    pub cam_tilt: f32,
    pub face_fill: f32,
    pub enter_dur: f32,
    pub dive_dur: f32,
    /// Skybox preset: "dusk" | "nebula" | "space" | "aurora".
    pub skybox: String,
    /// Floor reflection strength, 0 = off. ~0.4 is a subtle wet-floor look.
    pub reflection: f32,
    /// Floor surface color (linear rgb).
    pub floor_color: Vec3,
}

impl Default for CubeParams {
    fn default() -> Self {
        Self {
            board_recede: 1.5,
            base_float: 0.5,
            float_step: 0.38,
            overview_pullback: 0.7,
            cam_tilt: 0.16,
            face_fill: 0.80,
            enter_dur: 0.72,
            dive_dur: 0.55,
            skybox: "dusk".to_string(),
            reflection: 0.25,
            floor_color: Vec3::new(0.05, 0.06, 0.09),
        }
    }
}

const CUBE_RHAI_TEMPLATE: &str = "\
// editor-idea project prism (cube) tuning — hot-reloaded as you save.
// `board_recede` applies instantly; the rest apply next time you open
// the overview (Cmd+Shift+\\).
#{
    board_recede: 1.5,      // how far the background surface falls away
    base_float: 0.5,        // nearest pane's float off the surface
    float_step: 0.38,       // extra float per stacked pane
    overview_pullback: 0.7, // camera pull-back (x face height); lower = foreground barely shrinks
    cam_tilt: 0.16,         // overview downward tilt (x face height)
    face_fill: 0.80,        // fraction of the face the panes fit into
    enter_dur: 0.72,        // enter animation seconds
    dive_dur: 0.55,         // dive animation seconds
    skybox: \"dusk\",         // dusk | nebula | space | aurora
    reflection: 0.25,       // floor reflection strength (0 = off)
    floor_color: [0.05, 0.06, 0.09], // floor surface color, linear [r,g,b]
}
";

fn cube_rhai_path() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".jim/cube.rhai"))
}

fn rhai_f32(map: &rhai::Map, key: &str, default: f32) -> f32 {
    map.get(key)
        .and_then(|d| {
            d.as_float()
                .ok()
                .map(|v| v as f32)
                .or_else(|| d.as_int().ok().map(|v| v as f32))
        })
        .unwrap_or(default)
}

/// Read a 3-element rhai array (e.g. `[0.1, 0.2, 0.3]`) into a Vec3.
fn rhai_vec3(map: &rhai::Map, key: &str, default: Vec3) -> Vec3 {
    let Some(arr) = map.get(key).and_then(|d| d.clone().try_cast::<rhai::Array>()) else {
        return default;
    };
    let f = |i: usize, d: f32| {
        arr.get(i)
            .and_then(|x| {
                x.as_float()
                    .ok()
                    .map(|v| v as f32)
                    .or_else(|| x.as_int().ok().map(|v| v as f32))
            })
            .unwrap_or(d)
    };
    Vec3::new(f(0, default.x), f(1, default.y), f(2, default.z))
}

/// Poll `cube.rhai` for changes; re-evaluate and update `CubeParams` on
/// edit. Writes a default template on first run. Cheap (a stat per frame,
/// eval only on mtime change).
fn load_cube_params(
    mut params: ResMut<CubeParams>,
    mut last_mtime: Local<Option<std::time::SystemTime>>,
    mut initialized: Local<bool>,
) {
    let Some(path) = cube_rhai_path() else { return };
    if !*initialized {
        *initialized = true;
        if !path.exists() {
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let _ = std::fs::write(&path, CUBE_RHAI_TEMPLATE);
        }
    }
    let Ok(meta) = std::fs::metadata(&path) else { return };
    let Ok(mtime) = meta.modified() else { return };
    if *last_mtime == Some(mtime) {
        return;
    }
    *last_mtime = Some(mtime);
    let Ok(src) = std::fs::read_to_string(&path) else { return };
    let engine = rhai::Engine::new();
    match engine.eval::<rhai::Dynamic>(&src) {
        Ok(val) => {
            if let Ok(map) = val.try_cast::<rhai::Map>().ok_or(()) {
                let d = CubeParams::default();
                *params = CubeParams {
                    board_recede: rhai_f32(&map, "board_recede", d.board_recede),
                    base_float: rhai_f32(&map, "base_float", d.base_float),
                    float_step: rhai_f32(&map, "float_step", d.float_step),
                    overview_pullback: rhai_f32(&map, "overview_pullback", d.overview_pullback),
                    cam_tilt: rhai_f32(&map, "cam_tilt", d.cam_tilt),
                    face_fill: rhai_f32(&map, "face_fill", d.face_fill),
                    enter_dur: rhai_f32(&map, "enter_dur", d.enter_dur),
                    dive_dur: rhai_f32(&map, "dive_dur", d.dive_dur),
                    skybox: map
                        .get("skybox")
                        .and_then(|v| v.clone().into_string().ok())
                        .unwrap_or(d.skybox),
                    reflection: rhai_f32(&map, "reflection", d.reflection),
                    floor_color: rhai_vec3(&map, "floor_color", d.floor_color),
                };
                eprintln!("[cube] params reloaded: {:?}", *params);
            }
        }
        Err(e) => eprintln!("[cube] cube.rhai eval error: {e}"),
    }
}

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    #[default]
    Off,
    Entering,
    Overview,
    Diving,
    /// Dive finished and the new project is active, but the cube still
    /// covers the window for a few frames while the flat view settles
    /// (position + viewport catch up to the new active project), so the
    /// hand-off to flat has no mispositioned / sidebar-popping frame.
    Settling,
    /// Flat cameras/targets restored, but the (frozen) cube still covers
    /// the window for a frame or two so the per-pane cameras' window
    /// viewports settle invisibly — then we despawn the cube. Prevents the
    /// one-frame giant-pane flash at hand-off.
    Closing,
}

// ---------- State ----------

struct PaneFace {
    image: Handle<Image>,
    /// Dedicated camera that renders this pane's content into `image`.
    /// It ONLY ever targets `image` (fixed size), so — unlike retargeting
    /// the pane's own window camera — it never trips wgpu's "attachments
    /// have differing sizes" validation on a target switch.
    face_cam: Entity,
    quad: Entity,
    /// Reflection twin under `reflect_root` (shares the image/material).
    quad_reflect: Option<Entity>,
    project: u64,
    /// Local (root-relative) transform in the floating overview layout.
    cube_pose: Transform,
    /// Local transform that makes this pane coincide with exactly where
    /// the flat project view shows it (when the camera is at the
    /// face-fills-the-window pose). Morphing between the two makes the
    /// dive in/out a single seamless zoom — no cut to a different layout.
    flat_pose: Transform,
}

#[derive(Resource, Default)]
pub struct Prism {
    pub active: bool,
    pub pending_toggle: bool,
    phase: Phase,

    faces: HashMap<Entity, PaneFace>,
    /// (backboard, its reflection twin, project id) — picking + reflection.
    boards: Vec<(Entity, Option<Entity>, u64)>,
    /// project id -> face yaw angle (for dive/enter centering).
    face_angle: HashMap<u64, f32>,
    face_w: f32,

    root: Option<Entity>,
    cam: Option<Entity>,
    backdrop: Option<Entity>,
    /// Live handle to the sky material (so `animate_sky` can advance time).
    sky_mat: Option<Handle<SkyMaterial>>,
    /// Current skybox preset index (0..3). Tab cycles it live.
    sky_mode: f32,
    /// Frames to keep forcing continuous redraw AFTER exiting, so the flat
    /// panes (whose window cameras were off during the cube) get a full
    /// redraw instead of staying black until the user clicks.
    pub continuous_cooldown: u32,
    /// Mirror of `root` below the floor (holds the reflection twins).
    reflect_root: Option<Entity>,
    floor: Option<Entity>,
    floor_y: f32,
    disabled_cams: Vec<Entity>,
    /// Menu-overlay camera whose order we bump above the cube so the FPS
    /// meter / status bar stay visible over the overview; restored on exit.
    overlay_cam: Option<Entity>,
    prev_focus: Option<Entity>,

    yaw: f32,
    yaw_vel: f32,
    cam_dist: f32,
    cam_y: f32,
    overview_dist: f32,
    near_dist: f32,
    apothem: f32,

    // Animation (Entering / Diving).
    anim_t: f32,
    anim_dur: f32,
    start_yaw: f32,
    end_yaw: f32,
    start_dist: f32,
    end_dist: f32,
    start_cam_y: f32,
    end_cam_y: f32,
    dive_target: Option<u64>,
    /// The project whose panes morph between cube and flat layout during
    /// the current transition (the one we're flying into / out of).
    morph_project: Option<u64>,
    /// Frames remaining in the `Settling` phase before teardown.
    settle_frames: u32,

    // Input.
    dragging: bool,
    last_cursor_x: f32,
    press_pos: Option<Vec2>,
    pending_dive: Option<u64>,
}

// ---------- Plugin ----------

pub struct CubePlugin;

impl Plugin for CubePlugin {
    fn build(&self, app: &mut App) {
        bevy::asset::embedded_asset!(app, "sky.wgsl");
        bevy::asset::embedded_asset!(app, "floor.wgsl");
        app.add_plugins(MaterialPlugin::<SkyMaterial>::default());
        app.add_plugins(MaterialPlugin::<FloorMaterial>::default());
        app.init_resource::<Prism>()
            .init_resource::<CubeParams>()
            // While the overview covers the window, mouse input belongs to
            // the prism, not the panes/sidebar underneath. Consume it in
            // PreUpdate, before pane-bevy's `handle_pane_mouse` (Update)
            // reads the flag, so a rotate-drag never leaks into terminal
            // text selection. `InputConsumed` resets in PostUpdate, so this
            // re-arms every frame.
            .add_systems(PreUpdate, consume_input_during_overview)
            .add_systems(Update, (load_cube_params, animate_sky))
            // MUST run after `overview_apply`: that system despawns the cube
            // (root -> None) on the exit frame, and the cube camera is gone
            // by that frame's render. If suppress ran first it would leave
            // the pane window cameras off for one frame -> panes vanish.
            .add_systems(
                Update,
                (
                    overview_input,
                    overview_apply,
                    rotate_and_zoom_prism,
                    apply_prism_transform,
                )
                    .chain(),
            )
            .add_systems(
                Update,
                force_overview_state.after(crate::projects::sync_visibility),
            )
            // After `overview_apply` so the window cameras come back on the
            // same frame the cube despawns (no one-frame blank reveal).
            .add_systems(Update, suppress_window_pane_cams.after(overview_apply));
    }
}

// ---------- Input + picking ----------

/// While the overview is up (any phase but `Off`) the cube camera covers
/// the whole window, so the panes and sidebar behind it must not see the
/// mouse. Marking input consumed here gates every `!consumed` reader
/// (pane content press → terminal selection, pane drag/focus, sidebar
/// clicks) without each having to know about the cube. The prism's own
/// rotate/dive input reads the mouse buttons directly, so it's unaffected.
fn consume_input_during_overview(prism: Res<Prism>, mut consumed: ResMut<InputConsumed>) {
    if prism.phase != Phase::Off {
        consumed.0 = true;
    }
}

/// Detect the toggle / Escape / click-to-dive. Sets `pending_*` flags for
/// `overview_apply` to act on. Kept separate from `overview_apply` so the
/// read-only cube-camera query here doesn't conflict with that system's
/// `&mut Camera` access to the flat cameras.
fn overview_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut key_events: MessageReader<KeyboardInput>,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    owner: Res<pane_bevy::KeyboardOwner>,
    mut prism: ResMut<Prism>,
    cam_q: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    gt_q: Query<&GlobalTransform>,
) {
    // A text modal (command palette / rename) owns the keyboard — don't
    // toggle/drive the cube while typing.
    if owner.is_modal() {
        key_events.clear();
        return;
    }
    let cmd = keys.pressed(KeyCode::SuperLeft) || keys.pressed(KeyCode::SuperRight);
    let shift = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    for ev in key_events.read() {
        if !ev.state.is_pressed() {
            continue;
        }
        if cmd && shift && ev.key_code == KeyCode::Backslash {
            prism.pending_toggle = true;
        }
        if prism.active && ev.key_code == KeyCode::Escape {
            prism.pending_toggle = true;
        }
        // Tab cycles the skybox preset live (the shader reads the mode
        // uniform every frame, so no rebuild needed).
        if prism.active && ev.key_code == KeyCode::Tab {
            prism.sky_mode = (prism.sky_mode + 1.0) % 4.0;
        }
    }

    if prism.phase != Phase::Overview {
        return;
    }
    let Ok(window) = windows.single() else {
        return;
    };
    let cursor = window.cursor_position();

    if buttons.just_pressed(MouseButton::Left) {
        prism.press_pos = cursor;
    }
    if buttons.just_released(MouseButton::Left) {
        if let (Some(press), Some(now)) = (prism.press_pos.take(), cursor) {
            // A click (not a drag-rotate) dives into the project hit.
            if press.distance(now) < 6.0 {
                if let Ok((cam, cam_gt)) = cam_q.single() {
                    if let Some(pid) = pick_project(cam, cam_gt, now, &prism, &gt_q) {
                        prism.pending_dive = Some(pid);
                    }
                }
            }
        }
    }
}

/// Ray-pick the project face under the cursor. Ray vs each backboard
/// rectangle; nearest hit wins.
fn pick_project(
    cam: &Camera,
    cam_gt: &GlobalTransform,
    cursor: Vec2,
    prism: &Prism,
    gt_q: &Query<&GlobalTransform>,
) -> Option<u64> {
    let ray = cam.viewport_to_world(cam_gt, cursor).ok()?;
    let dir: Vec3 = ray.direction.into();
    let hw = prism.face_w * 0.5;
    let hh = FACE_HEIGHT * 0.5;

    let mut best: Option<(f32, u64)> = None;
    for (board, _twin, pid) in &prism.boards {
        let Ok(gt) = gt_q.get(*board) else { continue };
        let c = gt.translation();
        let rot = gt.rotation();
        let n = rot * Vec3::Z;
        let r = rot * Vec3::X;
        let u = rot * Vec3::Y;
        let denom = dir.dot(n);
        if denom.abs() < 1e-6 {
            continue;
        }
        let t = (c - ray.origin).dot(n) / denom;
        if t <= 0.0 {
            continue;
        }
        let p = ray.origin + dir * t;
        let d = p - c;
        if d.dot(r).abs() <= hw && d.dot(u).abs() <= hh {
            if best.map(|(bt, _)| t < bt).unwrap_or(true) {
                best = Some((t, *pid));
            }
        }
    }
    best.map(|(_, pid)| pid)
}

// ---------- Build / teardown / animation control ----------

type FlatCams<'w, 's> =
    Query<'w, 's, (Entity, &'static mut Camera), (With<Camera2d>, Without<PaneCameraOf>)>;

#[allow(clippy::too_many_arguments)]
fn overview_apply(
    time: Res<Time>,
    mut prism: ResMut<Prism>,
    mut projects: ResMut<Projects>,
    mut focus: ResMut<FocusedPane>,
    params: Res<CubeParams>,
    (canvas_view, pane_viewport, sidebar): (
        Res<crate::canvas::CanvasView>,
        Res<pane_bevy::PaneViewport>,
        Res<crate::projects::Sidebar>,
    ),
    windows: Query<&Window>,
    panes: Query<(Entity, &PaneRect, &PaneProject, Has<PanePinned>, &PaneLayer), With<PaneTag>>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    (mut sky_materials, mut floor_materials): (
        ResMut<Assets<SkyMaterial>>,
        ResMut<Assets<FloorMaterial>>,
    ),
    mut commands: Commands,
    mut winit: ResMut<bevy::winit::WinitSettings>,
    mut flat_cams: FlatCams,
    (style_state, preset_registry, style_data_dir): (
        Res<style_bevy::ProjectStyleState>,
        Res<style_bevy::presets::StylePresetRegistry>,
        Option<Res<style_bevy::StyleDataDir>>,
    ),
) {
    if prism.phase == Phase::Off && prism.continuous_cooldown > 0 {
        prism.continuous_cooldown -= 1;
    }
    // ----- Requests -----
    if std::mem::take(&mut prism.pending_toggle) {
        match prism.phase {
            Phase::Off => {
                if let Ok(window) = windows.single() {
                    // Force continuous redraw from frame 0 of the enter so
                    // the animation doesn't start in reactive (laggy) mode.
                    // Only `focused_mode` — `maintain_winit_mode_for_animation`
                    // owns reverting this when the overview closes.
                    winit.focused_mode = bevy::winit::UpdateMode::Continuous;
                    build(
                        &mut prism, &projects, &mut focus, &params, &canvas_view, &pane_viewport,
                        sidebar.width,
                        window, &panes, &mut images, &mut meshes, &mut materials,
                        &mut sky_materials, &mut floor_materials, &mut commands, &mut flat_cams,
                        &style_state, &preset_registry, style_data_dir.as_deref(),
                    );
                }
            }
            Phase::Overview => start_dive(&mut prism, &projects, None, params.dive_dur),
            _ => {}
        }
    }
    if let Some(pid) = prism.pending_dive.take() {
        if prism.phase == Phase::Overview {
            start_dive(&mut prism, &projects, Some(pid), params.dive_dur);
        }
    }

    // ----- Animation -----
    match prism.phase {
        Phase::Entering | Phase::Diving => {
            // Clamp dt: the first frame after toggling in can carry a big
            // reactive-mode gap, which would jump the animation forward and
            // read as a lag/stutter. Cap it so the start is smooth.
            let dt = time.delta_secs().min(1.0 / 30.0);
            prism.anim_t += dt / prism.anim_dur.max(1e-3);
            let e = smoothstep(prism.anim_t.clamp(0.0, 1.0));
            prism.yaw = lerp_angle(prism.start_yaw, prism.end_yaw, e);
            prism.cam_dist = lerp(prism.start_dist, prism.end_dist, e);
            prism.cam_y = lerp(prism.start_cam_y, prism.end_cam_y, e);
            if prism.anim_t >= 1.0 {
                if prism.phase == Phase::Entering {
                    prism.phase = Phase::Overview;
                } else {
                    // Dive complete: switch project now, but keep the cube
                    // covering the window for a few frames (Settling) so the
                    // flat view settles before we reveal it.
                    if let Some(target) = prism.dive_target {
                        projects.set_active(target);
                    }
                    prism.phase = Phase::Settling;
                    prism.settle_frames = 3;
                }
            }
        }
        Phase::Settling => {
            if prism.settle_frames == 0 {
                // Stage 1: restore flat cameras/targets/focus, but keep the
                // (frozen) cube covering the window so the per-pane camera
                // viewports settle invisibly.
                reset_to_flat(&mut prism, &mut flat_cams, &mut focus);
                prism.phase = Phase::Closing;
                prism.settle_frames = 1;
            } else {
                prism.settle_frames -= 1;
            }
        }
        Phase::Closing => {
            if prism.settle_frames == 0 {
                despawn_cube(&mut prism, &mut commands, &mut images);
            } else {
                prism.settle_frames -= 1;
            }
        }
        _ => {}
    }
}

fn start_dive(prism: &mut Prism, projects: &Projects, target: Option<u64>, dive_dur: f32) {
    // target = Some(pid): fly into that project. None: fly back into the
    // active project (Escape / close).
    let pid = target.or(projects.active);
    let end_yaw = pid
        .and_then(|p| prism.face_angle.get(&p).copied())
        .map(|theta| -theta)
        .unwrap_or(prism.yaw);
    prism.dive_target = target;
    prism.morph_project = pid;
    prism.start_yaw = prism.yaw;
    prism.end_yaw = end_yaw;
    prism.start_dist = prism.cam_dist;
    prism.end_dist = prism.near_dist;
    prism.start_cam_y = prism.cam_y;
    prism.end_cam_y = 0.0;
    prism.anim_t = 0.0;
    prism.anim_dur = dive_dur;
    prism.phase = Phase::Diving;
    prism.dragging = false;
    prism.yaw_vel = 0.0;
}

/// Background color a project's panes sit against, honoring its
/// per-project preset (or its own theme.rhai, or the built-in default).
/// Used to tint each prism face's backboard so the overview shows each
/// project in its real theme rather than the active project's. Loads the
/// theme file fresh, but only when the cube is built (occasional), so the
/// extra I/O is negligible.
fn project_theme_bg(
    pid: u64,
    style_state: &style_bevy::ProjectStyleState,
    preset_registry: &style_bevy::presets::StylePresetRegistry,
    style_data_dir: Option<&style_bevy::StyleDataDir>,
) -> Color {
    let path = style_state
        .preset_of(pid)
        .and_then(|name| {
            preset_registry
                .presets
                .iter()
                .find(|p| p.name == name)
                .map(|p| p.theme_path.clone())
        })
        .or_else(|| style_data_dir.map(|d| style_bevy::theme::theme_path_for_project(d, pid)));
    let bg = path
        .filter(|p| p.exists())
        .and_then(|p| style_bevy::theme::load_theme(&p).ok())
        .map(|t| t.color(style_bevy::tokens::BG))
        .unwrap_or_else(|| style_bevy::Theme::default().color(style_bevy::tokens::BG));
    Color::LinearRgba(bg)
}

#[allow(clippy::too_many_arguments)]
fn build(
    prism: &mut Prism,
    projects: &Projects,
    focus: &mut FocusedPane,
    params: &CubeParams,
    canvas_view: &crate::canvas::CanvasView,
    viewport: &pane_bevy::PaneViewport,
    sidebar_width: f32,
    window: &Window,
    panes: &Query<(Entity, &PaneRect, &PaneProject, Has<PanePinned>, &PaneLayer), With<PaneTag>>,
    images: &mut Assets<Image>,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    sky_materials: &mut Assets<SkyMaterial>,
    floor_materials: &mut Assets<FloorMaterial>,
    commands: &mut Commands,
    flat_cams: &mut FlatCams,
    style_state: &style_bevy::ProjectStyleState,
    preset_registry: &style_bevy::presets::StylePresetRegistry,
    style_data_dir: Option<&style_bevy::StyleDataDir>,
) {
    // Park keyboard focus so keys (Escape!) don't leak into terminals.
    prism.prev_focus = focus.0;
    focus.0 = None;

    // NOTE: we no longer deactivate the flat cameras. The cube camera
    // (order 1_000_000) clears and draws over the whole window, so they're
    // hidden anyway — and keeping them continuously active avoids the
    // one-frame stale-projection "jump" a just-reactivated camera renders
    // on exit.
    prism.disabled_cams.clear();

    // Lift the menu-overlay camera ABOVE the cube so the FPS meter / status
    // bar (which render on its layer) stay visible over the overview.
    prism.overlay_cam = None;
    for (e, mut cam) in flat_cams.iter_mut() {
        if cam.order == crate::MENU_OVERLAY_CAMERA_ORDER {
            cam.order = CUBE_CAM_ORDER + 1000;
            prism.overlay_cam = Some(e);
        }
    }

    let sf = window.scale_factor();
    let win_w = window.width();
    let win_h = window.height().max(1.0);
    let aspect = win_w / win_h;
    let face_w = FACE_HEIGHT * aspect;
    prism.face_w = face_w;

    // Every pane has a project (enforced invariant — see
    // `assert_pane_project_invariant`), so a pane lands on exactly its
    // own project's face. No `.or(active)` fallback: a membership-less
    // pane must not silently leak onto whatever happens to be active.
    //
    // Each pane is also isolated by its unique `PaneLayer` — the per-face
    // camera renders only that layer. Two live panes sharing a layer
    // would cross-render onto each other's faces (across projects), so
    // assert uniqueness here where the cube relies on it.
    let mut by_project: BTreeMap<u64, Vec<(Entity, PaneRect, bool, usize)>> = BTreeMap::new();
    let mut seen_layers: std::collections::HashMap<usize, Entity> = std::collections::HashMap::new();
    for (e, rect, proj, pinned, layer) in panes.iter() {
        if let Some(prev) = seen_layers.insert(layer.0, e) {
            panic!(
                "panes {prev:?} and {e:?} share render layer {} — per-pane render \
                 isolation is broken; the cube would cross-render them across projects",
                layer.0
            );
        }
        by_project.entry(proj.0).or_default().push((e, *rect, pinned, layer.0));
    }

    // Only switchable (non-parked) projects get a face — the prism is a
    // project switcher, and hidden projects are excluded from every
    // switcher by contract (see `Projects::switchable`).
    let mut ids: Vec<u64> = projects.switchable_ids();
    ids.sort_unstable();
    if ids.is_empty() {
        // No catalog entries — fall back to projects that have live panes,
        // still skipping any that are parked.
        ids = by_project
            .keys()
            .copied()
            .filter(|id| !projects.is_hidden(*id))
            .collect();
    }
    let n = ids.len().max(1);

    let apothem = if n >= 3 {
        (face_w * 0.5) / (std::f32::consts::PI / n as f32).tan()
    } else {
        face_w * 0.9
    };

    let root = commands
        .spawn((Transform::default(), Visibility::Visible, Name::new("prism_root")))
        .id();

    prism.face_angle.clear();
    prism.boards.clear();
    prism.faces.clear();

    // Reflection: a mirror of `root` below a translucent floor, holding
    // twin quads that share the live images. `apply_prism_transform`
    // keeps it the mirror of `root`.
    let reflect_on = params.reflection > 0.001;
    // Floor sits right at the base of the boards so the reflection meets
    // the prism (reads as "sitting on a surface", not floating below).
    let floor_y = -(FACE_HEIGHT * 0.5);
    prism.floor_y = floor_y;
    let reflect_root = if reflect_on {
        Some(
            commands
                .spawn((Transform::default(), Visibility::Visible, Name::new("prism_reflect")))
                .id(),
        )
    } else {
        None
    };
    prism.reflect_root = reflect_root;
    prism.floor = if reflect_on {
        let floor_mesh = meshes.add(Mesh::from(Rectangle::new(80.0, 80.0)));
        let floor_mat = floor_materials.add(FloorMaterial {
            params: params.floor_color.extend(params.reflection.clamp(0.0, 1.0)),
        });
        Some(
            commands
                .spawn((
                    Mesh3d(floor_mesh),
                    MeshMaterial3d(floor_mat),
                    Transform {
                        translation: Vec3::new(0.0, floor_y, 0.0),
                        rotation: Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
                        ..default()
                    },
                    render_layer(CUBE_LAYER),
                    Name::new("cube_floor"),
                ))
                .id(),
        )
    } else {
        None
    };

    for (i, &pid) in ids.iter().enumerate() {
        let theta = (i as f32 / n as f32) * std::f32::consts::TAU;
        prism.face_angle.insert(pid, theta);
        let dir = Vec3::new(theta.sin(), 0.0, theta.cos());
        let right = Vec3::new(theta.cos(), 0.0, -theta.sin());
        let up = Vec3::Y;
        let face_center = dir * apothem;
        let face_rot = Quat::from_rotation_y(theta);

        let board_mesh = meshes.add(Mesh::from(Rectangle::new(face_w, FACE_HEIGHT)));
        // Tint each face's backboard with THAT project's theme background,
        // not a hardcoded slate — so the overview shows every project in
        // its own theme instead of one theme across all faces.
        let board_mat = materials.add(StandardMaterial {
            base_color: project_theme_bg(pid, style_state, preset_registry, style_data_dir),
            unlit: true,
            cull_mode: None,
            double_sided: true,
            ..default()
        });
        // Initial pose only — `apply_prism_transform` overwrites this every
        // frame (incl. the first) with the rigid recede + recess. Keep it on
        // the apothem here so it's a clean n-gon even if never updated; the
        // inward recess is applied rigidly, never per-normal (see below).
        let board_pose = Transform {
            translation: face_center,
            rotation: face_rot,
            ..default()
        };
        let board = commands
            .spawn((
                Mesh3d(board_mesh.clone()),
                MeshMaterial3d(board_mat.clone()),
                board_pose,
                render_layer(CUBE_LAYER),
                Name::new(format!("face_board:{pid}")),
            ))
            .id();
        commands.entity(root).add_child(board);
        let board_twin = reflect_root.map(|rr| {
            let t = commands
                .spawn((
                    Mesh3d(board_mesh.clone()),
                    MeshMaterial3d(board_mat.clone()),
                    board_pose,
                    render_layer(CUBE_LAYER),
                    Name::new(format!("face_board_r:{pid}")),
                ))
                .id();
            commands.entity(rr).add_child(t);
            t
        });
        prism.boards.push((board, board_twin, pid));

        let Some(plist) = by_project.get(&pid) else {
            continue;
        };
        if plist.is_empty() {
            continue;
        }

        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        for (_, r, _, _) in plist {
            min = min.min(r.pos);
            max = max.max(r.pos + r.size);
        }
        let bbox = (max - min).max(Vec2::splat(1.0));
        let bbox_center = (min + max) * 0.5;
        let k = (face_w * params.face_fill / bbox.x).min(FACE_HEIGHT * params.face_fill / bbox.y);

        // This project's live pan/zoom — used to place panes exactly where
        // the flat view shows them, for the seamless dive morph.
        let view = canvas_view.state_for(pid);

        let mut ordered: Vec<&(Entity, PaneRect, bool, usize)> = plist.iter().collect();
        ordered.sort_by(|a, b| a.1.z.partial_cmp(&b.1.z).unwrap_or(std::cmp::Ordering::Equal));

        // Pinned panes are the anchored background — they lie flat on the
        // face (no float). Only unpinned panes pop out, stacked by z.
        let mut stack = 0usize;
        for (oi, (pane_e, rect, pinned, layer)) in ordered.iter().enumerate() {
            // Painter-order depth: every pane gets a unique tiny offset so
            // coplanar quads never z-fight.
            let eps = oi as f32 * DEPTH_EPS;
            let center = rect.pos + rect.size * 0.5;
            let u = (center.x - bbox_center.x) * k;
            let v = -(center.y - bbox_center.y) * k;
            let float = if *pinned {
                // Pinned panes lie near the face plane (just above the
                // recessed board), not popped out.
                PIN_LIFT + eps
            } else {
                let f = params.base_float + stack as f32 * params.float_step;
                stack += 1;
                f
            };
            let pos = face_center + right * u + up * v + dir * float;
            let world_size = rect.size * k;

            // Flat pose: where this pane sits when the project fills the
            // window at its own pan/zoom. Map the flat screen rect into the
            // face plane (the face exactly covers the window at the dive
            // endpoint), float 0.
            let screen_pos = (rect.pos - view.pan) * view.zoom + Vec2::new(sidebar_width, 0.0);
            let screen_size = rect.size * view.zoom;
            let screen_center = screen_pos + screen_size * 0.5;
            let fu = (screen_center.x / win_w - 0.5) * face_w;
            let fv = -(screen_center.y / win_h - 0.5) * FACE_HEIGHT;
            // `eps` (painter-order depth) keeps overlapping flat panes from
            // z-fighting at the dive endpoint and lifts them off the board.
            let flat_trans = face_center + right * fu + up * fv + dir * (PIN_LIFT + eps);
            let fsx = (screen_size.x / win_w) * face_w;
            let fsy = (screen_size.y / win_h) * FACE_HEIGHT;
            let flat_pose = Transform {
                translation: flat_trans,
                rotation: face_rot,
                scale: Vec3::new(
                    fsx / world_size.x.max(1e-4),
                    fsy / world_size.y.max(1e-4),
                    1.0,
                ),
            };
            let cube_pose = Transform {
                translation: pos,
                rotation: face_rot,
                ..default()
            };

            // Cap the texture resolution: panes are shown small in the
            // overview, so full native-res images are wasted memory. We
            // shrink the LONGEST side to MAX_FACE_TEX and fold the
            // reduction into the render-target scale factor so the face
            // camera still frames the WHOLE pane (just at lower res).
            let raw_w = (rect.size.x * sf).max(1.0);
            let raw_h = (rect.size.y * sf).max(1.0);
            let cap = (MAX_FACE_TEX / raw_w.max(raw_h)).min(1.0);
            let iw = (raw_w * cap).max(1.0) as u32;
            let ih = (raw_h * cap).max(1.0) as u32;
            let tex_sf = sf * cap;
            let mut img = Image::new_target_texture(iw, ih, TextureFormat::Bgra8UnormSrgb, None);
            // Render target: never read on the CPU, so don't keep a CPU copy
            // (halves the per-image footprint).
            img.asset_usage = bevy::asset::RenderAssetUsages::RENDER_WORLD;
            let image = images.add(img);

            // Dedicated camera that renders this pane's content (its render
            // layer) into `image`. We frame the pane WHERE IT ACTUALLY IS
            // (its flat world rect) rather than pinning it to the origin —
            // so the panes never move when the cube opens/closes, which is
            // what eliminates the one-frame "jump" at the exit. The camera
            // only ever targets `image`, so wgpu stays happy across toggles.
            let screen_tl = viewport.canvas_to_window(rect.pos);
            let world_tl = Vec2::new(screen_tl.x - win_w * 0.5, win_h * 0.5 - screen_tl.y);
            let pane_world_size = rect.size * viewport.zoom;
            let cam_center = Vec2::new(
                world_tl.x + pane_world_size.x * 0.5,
                world_tl.y - pane_world_size.y * 0.5,
            );
            let face_cam = commands
                .spawn((
                    Camera2d,
                    Camera {
                        viewport: Some(bevy::camera::Viewport {
                            physical_position: UVec2::ZERO,
                            physical_size: UVec2::new(iw, ih),
                            depth: 0.0..1.0,
                        }),
                        clear_color: ClearColorConfig::Custom(Color::NONE),
                        ..default()
                    },
                    // Match the live zoom so the framed world rect == the
                    // pane's actual on-screen rect.
                    Projection::from(OrthographicProjection {
                        scale: viewport.zoom.max(1e-4),
                        ..OrthographicProjection::default_2d()
                    }),
                    image_target(&image, tex_sf),
                    Transform::from_xyz(cam_center.x, cam_center.y, 0.0),
                    RenderLayers::layer(*layer),
                    // No MSAA: text/terminal content is already antialiased,
                    // and a 4x MSAA buffer per pane image is ~4x the texture
                    // memory (the bulk of the overview's footprint).
                    bevy::render::view::Msaa::Off,
                    Name::new(format!("face_cam:{pid}")),
                ))
                .id();

            let quad_mesh = meshes.add(Mesh::from(Rectangle::new(world_size.x, world_size.y)));
            let mat = materials.add(StandardMaterial {
                base_color: Color::WHITE,
                base_color_texture: Some(image.clone()),
                unlit: true,
                cull_mode: None,
                double_sided: true,
                // Alpha-MASK (not blend) so the quad writes depth — no
                // z-fighting with the board / other panes — while still
                // cutting out the pane's transparent rounded corners.
                alpha_mode: AlphaMode::Mask(0.5),
                ..default()
            });
            let quad = commands
                .spawn((
                    Mesh3d(quad_mesh.clone()),
                    MeshMaterial3d(mat.clone()),
                    cube_pose,
                    render_layer(CUBE_LAYER),
                    Name::new(format!("pane_face:{pid}:{:?}", pane_e)),
                ))
                .id();
            commands.entity(root).add_child(quad);
            let quad_reflect = reflect_root.map(|rr| {
                let t = commands
                    .spawn((
                        Mesh3d(quad_mesh.clone()),
                        MeshMaterial3d(mat.clone()),
                        cube_pose,
                        render_layer(CUBE_LAYER),
                        Name::new(format!("pane_face_r:{pid}")),
                    ))
                    .id();
                commands.entity(rr).add_child(t);
                t
            });
            prism.faces.insert(
                *pane_e,
                PaneFace {
                    image,
                    face_cam,
                    quad,
                    quad_reflect,
                    project: pid,
                    cube_pose,
                    flat_pose,
                },
            );
        }
    }

    // Dive/enter endpoint: the distance at which the face EXACTLY fills
    // the window (vertical fov, and face aspect == window aspect). The
    // pane morph lands on the flat layout here, so the hand-off to the
    // flat 2D view is seamless.
    let half_fov = (FOV_DEG * 0.5).to_radians();
    let near_dist = apothem + (FACE_HEIGHT * 0.5) / half_fov.tan();
    // Overview camera barely pulls back from the fill distance — the depth
    // reveal comes from the BOARDS receding, not the camera dollying, so
    // the foreground panes only get slightly smaller.
    let overview_dist = near_dist + FACE_HEIGHT * params.overview_pullback;
    let cam = commands
        .spawn((
            Camera3d::default(),
            Camera {
                order: CUBE_CAM_ORDER,
                clear_color: ClearColorConfig::Custom(BACKDROP),
                ..default()
            },
            Projection::from(PerspectiveProjection {
                fov: 45f32.to_radians(),
                // Tight near/far: a huge far plane wrecks depth precision and
                // makes the near-coplanar panes z-fight at the flat endpoint.
                // Far only needs to clear the skybox sphere (r=300) from the
                // camera's pulled-back position.
                near: 0.5,
                far: overview_dist + 340.0,
                ..default()
            }),
            Transform::from_xyz(0.0, 0.0, near_dist).looking_at(Vec3::ZERO, Vec3::Y),
            render_layer(CUBE_LAYER),
            Name::new("cube_cam"),
        ))
        .id();

    // Skybox: a huge inside-out sphere with the procedural `sky.wgsl`
    // shader. `params.x` selects the preset; `animate_sky` advances time.
    let sky_mesh = meshes.add(skybox_mesh());
    prism.sky_mode = skybox_mode(&params.skybox);
    let sky_mat = sky_materials.add(SkyMaterial {
        params: Vec4::new(prism.sky_mode, 0.0, 0.0, 0.0),
    });
    prism.sky_mat = Some(sky_mat.clone());
    let backdrop = commands
        .spawn((
            Mesh3d(sky_mesh),
            MeshMaterial3d(sky_mat),
            Transform::default(),
            render_layer(CUBE_LAYER),
            Name::new("cube_skybox"),
        ))
        .id();

    // Enter animation: dolly from near (at the active project's face) out
    // to the overview.
    let theta_active = projects
        .active
        .and_then(|p| prism.face_angle.get(&p).copied())
        .unwrap_or(0.0);
    prism.active = true;
    prism.phase = Phase::Entering;
    prism.root = Some(root);
    prism.cam = Some(cam);
    prism.backdrop = Some(backdrop);
    prism.overview_dist = overview_dist;
    prism.near_dist = near_dist;
    prism.apothem = apothem;
    prism.yaw = -theta_active;
    prism.yaw_vel = 0.0;
    prism.cam_dist = near_dist;
    prism.cam_y = 0.0;
    prism.dragging = false;
    prism.press_pos = None;
    prism.dive_target = None;
    // Enter morphs the project we came from: start in its flat layout
    // (matching the editor view), pull out to the floating cube.
    prism.morph_project = projects.active;
    // Enter pulls straight back from the flat project (no rotation): the
    // camera dollies out and the windows bloom off the surface. Head-on
    // start and end keeps it matching the flat view we came from.
    prism.start_yaw = -theta_active;
    prism.end_yaw = -theta_active;
    prism.start_dist = near_dist;
    prism.end_dist = overview_dist;
    prism.start_cam_y = 0.0;
    prism.end_cam_y = FACE_HEIGHT * params.cam_tilt;
    prism.anim_t = 0.0;
    prism.anim_dur = params.enter_dur;
}

/// Stage 1 of teardown: re-enable the flat window cameras and restore
/// focus, but DON'T despawn the cube yet. `active` goes false so the
/// per-frame drivers stop, while the frozen cube keeps covering the
/// window for a frame. The dedicated face cameras are separate from the
/// panes' own window cameras (which we never touched), so there's nothing
/// to "hand back".
fn reset_to_flat(prism: &mut Prism, flat_cams: &mut FlatCams, focus: &mut FocusedPane) {
    for e in prism.disabled_cams.drain(..) {
        if let Ok((_, mut cam)) = flat_cams.get_mut(e) {
            cam.is_active = true;
        }
    }
    // Restore the menu-overlay camera's normal order.
    if let Some(e) = prism.overlay_cam.take() {
        if let Ok((_, mut cam)) = flat_cams.get_mut(e) {
            cam.order = crate::MENU_OVERLAY_CAMERA_ORDER;
        }
    }
    // Restore focus only on a plain close; a dive changed the active
    // project, so let the project-change refocus pick the new target.
    if prism.dive_target.is_none() {
        focus.0 = prism.prev_focus;
    }
    prism.active = false;
}

/// Stage 2 of teardown: despawn the cube and free the per-pane images.
fn despawn_cube(prism: &mut Prism, commands: &mut Commands, images: &mut Assets<Image>) {
    if let Some(root) = prism.root.take() {
        commands.entity(root).despawn();
    }
    if let Some(cam) = prism.cam.take() {
        commands.entity(cam).despawn();
    }
    if let Some(backdrop) = prism.backdrop.take() {
        commands.entity(backdrop).despawn();
    }
    prism.sky_mat = None;
    if let Some(rr) = prism.reflect_root.take() {
        commands.entity(rr).despawn();
    }
    if let Some(floor) = prism.floor.take() {
        commands.entity(floor).despawn();
    }
    prism.boards.clear();
    // Despawning `root` cascades to its children (boards + quads), but the
    // dedicated face cameras are standalone — despawn them, and drop the
    // image assets.
    for (_, face) in prism.faces.drain() {
        commands.entity(face.face_cam).despawn();
        images.remove(&face.image);
    }
    prism.dive_target = None;
    prism.phase = Phase::Off;
    // Force a second of continuous redraw so the flat panes (window
    // cameras just re-enabled) actually get drawn before reactive mode
    // resumes — otherwise they stay black until clicked.
    prism.continuous_cooldown = 60;
}

// ---------- Per-frame drivers ----------

/// Single authority for whether each pane's OWN window camera renders.
/// A pane camera is active IFF the cube is down AND the pane belongs to
/// the active project.
///
/// This is what STRUCTURALLY confines a pane to its project. The old
/// behaviour left every flat camera active whenever the cube was down —
/// every project's pane content was being rendered to the window, and
/// non-active projects were blanked only by each content entity
/// inheriting the pane's `Visibility::Hidden`. That made confinement
/// depend on every pane kind spawning its content with the right
/// visibility (the terminal grid and widgets do; the editor's text did
/// not — so editor panes leaked across projects). Gating the CAMERA
/// instead means content can never render outside its project no matter
/// how a kind spawns it: the only camera that draws a pane's render
/// layer is off unless that pane is in the active project.
///
/// While the cube is up, ALL flat cameras are off — the dedicated face
/// cameras render the overview, and leaving these on would double the
/// per-pane render-to-texture cost.
///
/// MUST run after `overview_apply` (see registration): that's the system
/// that despawns the cube; if we ran first we'd leave them off for one
/// frame at the reveal and the panes would vanish.
fn suppress_window_pane_cams(
    prism: Res<Prism>,
    projects: Res<Projects>,
    panes: Query<&PaneProject>,
    mut cams: Query<(&PaneCameraOf, &mut Camera)>,
) {
    let cube_up = prism.root.is_some();
    let active = projects.active;
    for (owner, mut cam) in &mut cams {
        // No fallback: a pane whose project isn't the active one (or that
        // somehow has no membership) does not render. Membership is an
        // enforced invariant — see `assert_pane_project_invariant`.
        let want_active = !cube_up
            && panes
                .get(owner.0)
                .map(|p| Some(p.0) == active)
                .unwrap_or(false);
        if cam.is_active != want_active {
            cam.is_active = want_active;
        }
    }
}

/// Advance the sky shader's time uniform so the nebula drifts and stars
/// twinkle. Cheap; only touches the one material while the prism is up.
fn animate_sky(
    prism: Res<Prism>,
    time: Res<Time>,
    mut sky_materials: ResMut<Assets<SkyMaterial>>,
) {
    if !prism.active {
        return;
    }
    if let Some(handle) = &prism.sky_mat {
        if let Some(mat) = sky_materials.get_mut(handle) {
            mat.params.x = prism.sky_mode;
            mat.params.y = time.elapsed_secs();
        }
    }
}

/// While the overview is up: keep every project's panes visible (override
/// `sync_visibility`) and keep keyboard focus parked off all panes.
fn force_overview_state(
    prism: Res<Prism>,
    mut focus: ResMut<FocusedPane>,
    mut panes: Query<&mut Visibility, With<PaneTag>>,
) {
    if !prism.active {
        return;
    }
    if focus.0.is_some() {
        focus.0 = None;
    }
    for mut vis in &mut panes {
        if *vis != Visibility::Visible {
            *vis = Visibility::Visible;
        }
    }
}

// ---------- Rotate + zoom (Overview only) ----------

fn rotate_and_zoom_prism(
    mut prism: ResMut<Prism>,
    time: Res<Time>,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    mut wheel: MessageReader<MouseWheel>,
) {
    if prism.phase != Phase::Overview {
        wheel.clear();
        return;
    }
    let Ok(window) = windows.single() else {
        return;
    };
    let cursor_x = window.cursor_position().map(|p| p.x);

    if buttons.just_pressed(MouseButton::Left) {
        if let Some(x) = cursor_x {
            prism.dragging = true;
            prism.last_cursor_x = x;
            prism.yaw_vel = 0.0;
        }
    }
    if buttons.just_released(MouseButton::Left) {
        prism.dragging = false;
    }
    if prism.dragging {
        if let Some(x) = cursor_x {
            let dx = x - prism.last_cursor_x;
            prism.last_cursor_x = x;
            let dyaw = dx * 0.006;
            prism.yaw += dyaw;
            let dt = time.delta_secs().max(1e-4);
            prism.yaw_vel = dyaw / dt;
        }
    } else {
        prism.yaw += prism.yaw_vel * time.delta_secs();
        prism.yaw_vel *= 0.90;
        if prism.yaw_vel.abs() < 0.01 {
            prism.yaw_vel = 0.0;
        }
    }

    let mut scroll = 0.0;
    for ev in wheel.read() {
        let s = match ev.unit {
            MouseScrollUnit::Line => 1.0,
            MouseScrollUnit::Pixel => 1.0 / 16.0,
        };
        scroll += ev.y * s;
    }
    if scroll != 0.0 {
        prism.cam_dist = (prism.cam_dist * (1.0 - scroll * 0.08)).clamp(2.0, 400.0);
    }
}

/// Apply yaw + cam pose to the prism root and camera, and morph each
/// pane quad between its floating cube pose and its flat-layout pose so a
/// dive in/out is one continuous zoom.
fn apply_prism_transform(
    prism: Res<Prism>,
    params: Res<CubeParams>,
    mut roots: Query<&mut Transform, (Without<Camera3d>, Without<PaneTag>)>,
    mut cams: Query<&mut Transform, With<Camera3d>>,
) {
    if !prism.active {
        return;
    }
    if let Some(root) = prism.root {
        if let Ok(mut t) = roots.get_mut(root) {
            t.rotation = Quat::from_rotation_y(prism.yaw);
        }
    }
    if let Some(cam) = prism.cam {
        if let Ok(mut t) = cams.get_mut(cam) {
            *t = Transform::from_xyz(0.0, prism.cam_y, prism.cam_dist)
                .looking_at(Vec3::ZERO, Vec3::Y);
        }
    }
    // Keep the reflection root the mirror of `root` across the floor: same
    // yaw, scaled -1 in Y, translated so it sits below y = floor_y. With
    // twin locals == real locals, every twin lands at the mirror of its
    // real quad automatically. (A Y-rotation commutes with the y-flip, so
    // the same yaw + scale.y=-1 is a correct mirror.)
    if let Some(rr) = prism.reflect_root {
        if let Ok(mut t) = roots.get_mut(rr) {
            *t = Transform {
                translation: Vec3::new(0.0, 2.0 * prism.floor_y, 0.0),
                rotation: Quat::from_rotation_y(prism.yaw),
                scale: Vec3::new(1.0, -1.0, 1.0),
            };
        }
    }

    // How "cube-like" the morphing project's panes are right now:
    // 1.0 = fully floating cube, 0.0 = flat layout.
    let e = smoothstep(prism.anim_t.clamp(0.0, 1.0));
    let morph_cube_frac = match prism.phase {
        Phase::Entering => e,
        Phase::Diving => 1.0 - e,
        // Hold the flat layout while the flat view settles behind us.
        Phase::Settling => 0.0,
        _ => 1.0,
    };
    for face in prism.faces.values() {
        let frac = if Some(face.project) == prism.morph_project {
            morph_cube_frac
        } else {
            1.0
        };
        let pose = lerp_transform(&face.flat_pose, &face.cube_pose, frac);
        if let Ok(mut t) = roots.get_mut(face.quad) {
            *t = pose;
        }
        if let Some(twin) = face.quad_reflect {
            if let Ok(mut t) = roots.get_mut(twin) {
                *t = pose;
            }
        }
    }

    // Recede the backboards: they fall away from the camera as the
    // overview opens (and come back to meet the panes at the flat
    // endpoint of a dive). This is the "background moving away" effect —
    // the panes stay near the camera, the surface drops back.
    for (board, twin, pid) in &prism.boards {
        let frac = if Some(*pid) == prism.morph_project {
            morph_cube_frac
        } else {
            1.0
        };
        let theta = prism.face_angle.get(pid).copied().unwrap_or(0.0);
        let dir = Vec3::new(theta.sin(), 0.0, theta.cos());
        let face_center = dir * prism.apothem;
        // Recede the WHOLE n-gon RIGIDLY away from the camera (world -Z),
        // not each board inward along its own normal (which collapsed the
        // boards into an X). Pre-rotating by -yaw means that after the
        // root's yaw rotation the net shift is straight back from the
        // camera, so the n-gon keeps its clean shape and just sits behind
        // the (un-shifted) panes.
        // Recede straight back from the camera (world -Z after the root
        // yaw). BOTH the animated recede AND the constant board offset go
        // in here. Applying the constant offset per-board-normal instead
        // (`- dir * BOARD_RECESS`) pulls each board toward the axis, which
        // shrinks its effective radius below the apothem and makes adjacent
        // boards interpenetrate — the same "collapse into an X" noted above,
        // just small enough to only show when the animated recede is near
        // zero (mid open/dive/morph). Folding it into the rigid shift keeps
        // the n-gon a clean prism at every animation frame.
        let back = Quat::from_rotation_y(-prism.yaw)
            * Vec3::new(0.0, 0.0, -(params.board_recede * frac + BOARD_RECESS));
        let pose = Transform {
            translation: face_center + back,
            rotation: Quat::from_rotation_y(theta),
            ..default()
        };
        if let Ok(mut t) = roots.get_mut(*board) {
            *t = pose;
        }
        if let Some(tw) = twin {
            if let Ok(mut t) = roots.get_mut(*tw) {
                *t = pose;
            }
        }
    }
}

// ---------- Helpers ----------

fn image_target(image: &Handle<Image>, scale_factor: f32) -> RenderTarget {
    RenderTarget::Image(bevy::camera::ImageRenderTarget {
        handle: image.clone(),
        scale_factor,
    })
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn lerp_angle(a: f32, b: f32, t: f32) -> f32 {
    let mut d = (b - a) % std::f32::consts::TAU;
    if d > std::f32::consts::PI {
        d -= std::f32::consts::TAU;
    } else if d < -std::f32::consts::PI {
        d += std::f32::consts::TAU;
    }
    a + d * t
}

fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Lerp from `a` to `b` by `t` (translation + scale lerp, rotation slerp).
fn lerp_transform(a: &Transform, b: &Transform, t: f32) -> Transform {
    Transform {
        translation: a.translation.lerp(b.translation, t),
        rotation: a.rotation.slerp(b.rotation, t),
        scale: a.scale.lerp(b.scale, t),
    }
}

/// A large sphere, flipped inside-out (reversed winding) so its inner
/// surface is front-facing — we view the procedural sky from inside it.
fn skybox_mesh() -> Mesh {
    let mut mesh = Mesh::from(Sphere::new(300.0));
    if let Some(indices) = mesh.indices_mut() {
        match indices {
            bevy::mesh::Indices::U16(v) => {
                for tri in v.chunks_exact_mut(3) {
                    tri.swap(1, 2);
                }
            }
            bevy::mesh::Indices::U32(v) => {
                for tri in v.chunks_exact_mut(3) {
                    tri.swap(1, 2);
                }
            }
        }
    }
    mesh
}
