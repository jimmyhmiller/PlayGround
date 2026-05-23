//! Shader data providers + uniform-block plumbing.
//!
//! The host registers data sources ("providers") that produce named
//! values; shaders read them through fixed-layout uniform blocks
//! declared in the WGSL prelude. For v0 the uniform structs are
//! hand-defined in Rust and the WGSL prelude is hand-shipped to match —
//! the provider abstraction is preserved so future tiers can be added
//! by registration alone, but actual codegen of WGSL structs from the
//! field registry is deferred.
//!
//! ## Three scopes
//!
//! - **World**: the same for every shader — time, mouse, focused pane.
//! - **Project**: per active project — dust, last-edit, age.
//! - **Pane**: per pane (not used by the canvas-background shader, but
//!   reserved so per-pane shaders can be added later).

use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;

use crate::dev::DevOverrides;
use crate::state::ProjectStyleState;
use crate::theme::{tokens, Theme, TokenValue};

/// Side length (square) of the per-project wipe mask texture. UV-mapped
/// across the whole window — i.e. a logical 1200×800 canvas samples
/// the same 1024×1024 mask in normalized coords. Larger = smoother
/// smears but more upload bandwidth on each mouse move. 1024 is plenty
/// for the kind of casual cursor smearing we expect.
///
/// Brush radius and the dust-gate threshold are live-tunable via the
/// theme tokens `wipe_brush_radius_px` and `wipe_dust_gate_secs`.
pub const WIPE_MASK_SIZE: u32 = 1024;

/// Currently the host tells style-bevy which project is "active" via
/// this resource. The canvas background uses the active project's
/// state to compute Tier-2 uniforms.
#[derive(Resource, Default, Clone, Copy, Debug)]
pub struct ActiveProject(pub Option<u64>);

// ---------- Field metadata (used for introspection / future codegen) ----------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldType {
    F32,
    Vec2,
    Vec4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldScope {
    World,
    Project,
    Pane,
}

#[derive(Clone, Copy, Debug)]
pub struct ShaderField {
    pub name: &'static str,
    pub ty: FieldType,
    pub scope: FieldScope,
}

#[derive(Clone, Copy, Debug)]
pub enum FieldValue {
    F32(f32),
    Vec2(Vec2),
    Vec4(Vec4),
}

// ---------- Uniform buffer structs (match prelude.wgsl) ----------

/// Tier-0/1 fields. Matches `struct WorldData` in `prelude.wgsl`.
#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct WorldUniforms {
    pub time: f32,
    pub camera_zoom: f32,
    pub resolution: Vec2,
    pub mouse_world: Vec2,
    pub _pad: Vec2,
    pub focused_pane: Vec4,
    /// 0..1 progress of the current Windex animation; 0 when idle.
    /// Drives the spray + wipe sweep entirely in the shader so the
    /// animation can be tuned/redesigned by editing the WGSL on disk.
    pub windex_progress: f32,
    /// 1 while a Windex animation is playing, 0 otherwise. Shaders
    /// check this before doing any windex-specific work.
    pub windex_active: u32,
    pub _pad2: Vec2,
}

/// Tier-2 (per active project) fields. Matches `struct ProjectData`.
#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct ProjectUniforms {
    pub dust_seconds: f32,
    pub last_edit_seconds: f32,
    pub age_seconds: f32,
    pub _pad: f32,
}

/// Engine-known theme tokens, packed for shader use. Matches
/// `struct ThemeData`.
#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct ThemeUniforms {
    pub bg: Vec4,
    pub fg: Vec4,
    pub fg_muted: Vec4,
    pub accent: Vec4,
    pub caret: Vec4,
    pub selection: Vec4,
    pub warn: Vec4,
    pub err: Vec4,
    pub font_size: f32,
    pub line_height_ratio: f32,
    pub dust_intensity: f32,
    pub _pad: f32,
}

/// Holds the latest computed values for every uniform block, plus
/// registered field metadata for introspection. Providers write into
/// the typed structs each frame; the background-material system copies
/// from here into the material's bind group.
#[derive(Resource, Default)]
pub struct ShaderDataRegistry {
    pub world: WorldUniforms,
    pub project: ProjectUniforms,
    pub theme: ThemeUniforms,
    pub fields: Vec<ShaderField>,
}

impl ShaderDataRegistry {
    pub fn register(&mut self, field: ShaderField) {
        if self.fields.iter().any(|f| f.name == field.name) {
            return;
        }
        self.fields.push(field);
    }
}

// ---------- AppExt for terse provider registration ----------

/// Pairs a field descriptor with the system that fills it. Real
/// codegen-of-WGSL would consume `fields` to emit the prelude; for v0
/// the prelude is hand-shipped and we just keep the registry list as
/// documentation + future hook.
pub trait ShaderFieldsAppExt {
    fn register_shader_field(&mut self, field: ShaderField) -> &mut Self;
}

impl ShaderFieldsAppExt for App {
    fn register_shader_field(&mut self, field: ShaderField) -> &mut Self {
        self.world_mut()
            .resource_mut::<ShaderDataRegistry>()
            .register(field);
        self
    }
}

// ---------- Built-in providers (Tier 0/1/2) ----------

pub struct ShaderProviderPlugin;

/// Total time (seconds) the Windex animation runs end-to-end. Tunable
/// here (Rust) but the *visual* shape of the animation is entirely in
/// `default_background.wgsl` and is live-editable.
pub const WINDEX_DURATION: f32 = 1.6;

/// Active Windex animation, if any. `Some(t)` means a windex was
/// triggered at world.time = t; the animation ends when
/// `world.time - t >= WINDEX_DURATION`.
///
/// `just_completed` is a one-frame flag set by `tick_windex` when the
/// animation finishes; downstream systems consume it to do end-of-
/// animation work (clearing wipe masks, dev overrides, etc.) and
/// reset it.
#[derive(Resource, Default, Clone, Debug)]
pub struct WindexAnim {
    pub started_at: Option<f32>,
    pub just_completed: bool,
}

impl Plugin for ShaderProviderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ActiveProject>()
            .init_resource::<WindexAnim>()
            .add_systems(
                Update,
                (
                    tick_world_time,
                    tick_world_window,
                    tick_world_focus,
                    tick_windex,
                    tick_project,
                    tick_theme,
                ),
            );

        for field in built_in_fields() {
            app.register_shader_field(field);
        }
    }
}

/// Translate the live `WindexAnim` resource into shader uniforms each
/// frame. The shader sees a clean 0..1 progress value and a boolean
/// "is the animation running"; nothing else.
///
/// Also: when the animation completes this frame, set
/// `WindexAnim.just_completed = true` so the reset system can do
/// its end-of-animation work (clear mask + clear overrides + reset
/// `last_focus_at`). The flag is consumed on the next frame.
///
/// Also: drives `WinitSettings::focused_mode` so the window renders
/// continuously while the animation plays. Without this, reactive
/// mode means the animation only progresses when the user moves the
/// mouse — the user observed exactly this. Restores the host's
/// previously-configured `focused_mode` when the animation ends.
pub fn tick_windex(
    time: Res<Time>,
    mut anim: ResMut<WindexAnim>,
    mut reg: ResMut<ShaderDataRegistry>,
    mut settings: ResMut<bevy::winit::WinitSettings>,
    mut saved_mode: Local<Option<bevy::winit::UpdateMode>>,
) {
    if let Some(started_at) = anim.started_at {
        let elapsed = time.elapsed_secs() - started_at;
        if saved_mode.is_none() {
            // Snapshot the host's mode on the first frame of the
            // animation, then force Continuous so the animation
            // progresses without input events. Restored at end.
            *saved_mode = Some(settings.focused_mode);
            settings.focused_mode = bevy::winit::UpdateMode::Continuous;
        }
        if elapsed >= WINDEX_DURATION {
            anim.started_at = None;
            anim.just_completed = true;
            reg.world.windex_progress = 0.0;
            reg.world.windex_active = 0;
            if let Some(old) = saved_mode.take() {
                settings.focused_mode = old;
            }
        } else {
            reg.world.windex_progress = (elapsed / WINDEX_DURATION).clamp(0.0, 1.0);
            reg.world.windex_active = 1;
        }
    } else {
        reg.world.windex_progress = 0.0;
        reg.world.windex_active = 0;
        // Defensive: if anim was cancelled externally, restore mode.
        if let Some(old) = saved_mode.take() {
            settings.focused_mode = old;
        }
    }
}

fn built_in_fields() -> Vec<ShaderField> {
    use FieldScope::*;
    use FieldType::*;
    vec![
        ShaderField { name: "time", ty: F32, scope: World },
        ShaderField { name: "camera_zoom", ty: F32, scope: World },
        ShaderField { name: "resolution", ty: Vec2, scope: World },
        ShaderField { name: "mouse_world", ty: Vec2, scope: World },
        ShaderField { name: "focused_pane", ty: FieldType::Vec4, scope: World },
        ShaderField { name: "dust_seconds", ty: F32, scope: Project },
        ShaderField { name: "last_edit_seconds", ty: F32, scope: Project },
        ShaderField { name: "age_seconds", ty: F32, scope: Project },
    ]
}

fn tick_world_time(
    mut reg: ResMut<ShaderDataRegistry>,
    time: Res<Time>,
    overrides: Res<DevOverrides>,
) {
    let scale = overrides.time_scale.unwrap_or(1.0);
    reg.world.time = time.elapsed_secs() * scale;
}

fn tick_world_window(
    mut reg: ResMut<ShaderDataRegistry>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
) {
    if let Ok(window) = windows.single() {
        // Logical pixels (not physical) so resolution and mouse_world
        // share the same coordinate space. The dust quad is also
        // scaled to logical width/height by `resize_background_quad`,
        // so `uv * resolution` in the shader lands on the same pixel
        // grid as `mouse_world`.
        let w = window.width();
        let h = window.height();
        reg.world.resolution = Vec2::new(w, h);
        if let Some(cursor) = window.cursor_position() {
            reg.world.mouse_world = Vec2::new(cursor.x, cursor.y);
        }
    }
}

fn tick_world_focus(
    mut reg: ResMut<ShaderDataRegistry>,
    focused: Res<pane_bevy::FocusedPane>,
    rects: Query<&pane_bevy::PaneRect>,
) {
    let r = focused
        .0
        .and_then(|e| rects.get(e).ok())
        .map(|r| Vec4::new(r.pos.x, r.pos.y, r.size.x, r.size.y))
        .unwrap_or(Vec4::ZERO);
    reg.world.focused_pane = r;
    // camera_zoom is set by the host if it has its own camera; default
    // to 1.0 so shaders that read it don't see a NaN/0.
    if reg.world.camera_zoom == 0.0 {
        reg.world.camera_zoom = 1.0;
    }
}

pub fn tick_project(
    mut reg: ResMut<ShaderDataRegistry>,
    active: Res<ActiveProject>,
    state: Res<ProjectStyleState>,
    overrides: Res<DevOverrides>,
) {
    let Some(pid) = active.0 else {
        reg.project = ProjectUniforms::default();
        return;
    };
    let entry = state.entry(pid);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    let since = |t: f64| (now - t).max(0.0) as f32;
    let dust = if entry.last_focus_at == 0.0 { 0.0 } else { since(entry.last_focus_at) };
    let edit = if entry.last_edit_at == 0.0 { 0.0 } else { since(entry.last_edit_at) };
    let age = if entry.created_at == 0.0 { 0.0 } else { since(entry.created_at) };
    let proj_ov = overrides.project(pid);
    reg.project = ProjectUniforms {
        dust_seconds: proj_ov.dust_seconds.unwrap_or(dust),
        last_edit_seconds: proj_ov.last_edit_seconds.unwrap_or(edit),
        age_seconds: proj_ov.age_seconds.unwrap_or(age),
        _pad: 0.0,
    };
}

fn tick_theme(mut reg: ResMut<ShaderDataRegistry>, theme: Res<Theme>) {
    let col = |id| {
        let c = theme.color(id);
        Vec4::new(c.red, c.green, c.blue, c.alpha)
    };
    let dust_intensity = match theme.get(tokens::DUST_INTENSITY) {
        Some(TokenValue::F32(v)) => v,
        _ => 1.0,
    };
    reg.theme = ThemeUniforms {
        bg: col(tokens::BG),
        fg: col(tokens::FG),
        fg_muted: col(tokens::FG_MUTED),
        accent: col(tokens::ACCENT),
        caret: col(tokens::CARET),
        selection: col(tokens::SELECTION),
        warn: col(tokens::WARN),
        err: col(tokens::ERR),
        font_size: theme.f32(tokens::FONT_SIZE),
        line_height_ratio: theme.f32(tokens::LINE_HEIGHT_RATIO),
        dust_intensity,
        _pad: 0.0,
    };
}
