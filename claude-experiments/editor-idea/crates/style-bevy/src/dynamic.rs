//! Dynamic-schema material — the runtime side of the introspection
//! architecture.
//!
//! Where [`crate::introspect`] *learns* a shader's data shape,
//! [`DynamicMaterial`] *holds* the bytes the shader will read.
//!
//! ## Layout (fixed, never changes per shader)
//!
//! - `@group(M) @binding(0)`: a 2 KiB opaque uniform buffer
//!   (`UserBuffer = [Vec4; 128]`). The shader carves whatever struct
//!   it wants into the first N bytes; the host packs bytes by offset
//!   using the introspected `Schema`.
//! - `@group(M) @binding(1)`: a shared linear sampler for every
//!   texture in the bind group.
//! - `@group(M) @binding(2..)`: up to 8 `texture_2d<f32>` slots. The
//!   shader names whichever ones it wants in this range; the host
//!   maps name → slot index via the introspected `Schema.textures`.
//!
//! ## What a shader looks like
//!
//! ```wgsl
//! struct UserData { time: f32, dust_seconds: f32, ... }
//! @group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> user: UserData;
//! @group(#{MATERIAL_BIND_GROUP}) @binding(1) var samp: sampler;
//! @group(#{MATERIAL_BIND_GROUP}) @binding(2) var wipe_mask: texture_2d<f32>;
//! ```
//!
//! The shader's `samp` and `wipe_mask` names live in `Schema`; scripts
//! can refer to them by name (e.g. `mask_paint("wipe_mask", ...)`).

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dKey};

use crate::introspect::Schema;

/// Maximum bytes a shader's UserData struct may occupy. Bumping this
/// requires no logic change — just more memory per material. 2 KiB =
/// 128 `vec4<f32>` slots is more than enough for the kinds of effects
/// we're building (dust uses well under 200 bytes).
pub const USER_BUFFER_BYTES: usize = 2048;

/// Number of named texture slots the material exposes. The shader can
/// declare textures at any of bindings 2..=2+MAX_TEXTURES; whichever
/// ones it uses get their names recorded in the schema. Unused slots
/// hold a tiny dummy image so the bind group is always complete.
pub const MAX_TEXTURES: usize = 8;

/// Opaque 2 KiB chunk of bytes. Encase sees this as a struct of 128
/// `vec4<f32>` for std140 layout purposes — that's the largest
/// element alignment we'll need, so any sub-struct laid out by naga
/// will fit cleanly inside.
#[repr(C)]
#[derive(Clone, Copy, Debug, ShaderType)]
pub struct UserBuffer {
    pub slots: [Vec4; USER_BUFFER_BYTES / 16],
}

impl Default for UserBuffer {
    fn default() -> Self {
        Self {
            slots: [Vec4::ZERO; USER_BUFFER_BYTES / 16],
        }
    }
}

impl UserBuffer {
    /// View the buffer as a flat byte slice for `Schema::write_*`.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: UserBuffer is `repr(C)` and consists entirely of
        // `Vec4`s, which are themselves four contiguous f32 values
        // (no padding). The total size is exactly USER_BUFFER_BYTES.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.slots.as_mut_ptr() as *mut u8,
                USER_BUFFER_BYTES,
            )
        }
    }
}

/// Per-shader schema cache. Populated when a shader is loaded /
/// hot-reloaded; consulted by every host fn that mutates a material
/// (`uniform_set`, `mask_*`, ...). Wrapped in `RwLock` so the asset
/// reload path can replace entries while live systems read them.
#[derive(Resource, Default, Clone)]
pub struct ShaderSchemas {
    inner: Arc<RwLock<HashMap<AssetId<Shader>, Schema>>>,
}

impl ShaderSchemas {
    pub fn get(&self, id: AssetId<Shader>) -> Option<Schema> {
        self.inner.read().ok()?.get(&id).cloned()
    }
    pub fn set(&self, id: AssetId<Shader>, schema: Schema) {
        if let Ok(mut g) = self.inner.write() {
            g.insert(id, schema);
        }
    }
}

/// The dynamic Material2d. One per canvas (and conceptually one per
/// shader instance — same material, different shader handle).
#[derive(Asset, TypePath, AsBindGroup, Clone)]
#[bind_group_data(DynamicMaterialKey)]
pub struct DynamicMaterial {
    #[uniform(0)]
    pub user: UserBuffer,
    /// First texture slot ALSO carries the shared sampler attribute.
    /// In AsBindGroup, `#[sampler(N)]` must be attached to a texture
    /// field; the sampler ends up at its own binding N independently
    /// of where the texture itself sits.
    #[texture(2)]
    #[sampler(1)]
    pub tex0: Handle<Image>,
    #[texture(3)]
    pub tex1: Handle<Image>,
    #[texture(4)]
    pub tex2: Handle<Image>,
    #[texture(5)]
    pub tex3: Handle<Image>,
    #[texture(6)]
    pub tex4: Handle<Image>,
    #[texture(7)]
    pub tex5: Handle<Image>,
    #[texture(8)]
    pub tex6: Handle<Image>,
    #[texture(9)]
    pub tex7: Handle<Image>,
    /// Active fragment shader. Lives in the material so the host can
    /// hot-swap it per project; pipeline cache keys by the handle via
    /// `DynamicMaterialKey`.
    pub fragment: Handle<Shader>,
}

impl DynamicMaterial {
    /// Write a handle into the texture slot at `binding`. Bindings 2..=9
    /// map to `tex0..=tex7`.
    pub fn set_texture(&mut self, binding: u32, handle: Handle<Image>) {
        match binding {
            2 => self.tex0 = handle,
            3 => self.tex1 = handle,
            4 => self.tex2 = handle,
            5 => self.tex3 = handle,
            6 => self.tex4 = handle,
            7 => self.tex5 = handle,
            8 => self.tex6 = handle,
            9 => self.tex7 = handle,
            _ => eprintln!("[dynamic] texture binding {} out of range (2..=9)", binding),
        }
    }
}

#[derive(Hash, PartialEq, Eq, Clone)]
pub struct DynamicMaterialKey {
    pub fragment: Handle<Shader>,
}

impl From<&DynamicMaterial> for DynamicMaterialKey {
    fn from(m: &DynamicMaterial) -> Self {
        Self {
            fragment: m.fragment.clone(),
        }
    }
}

impl Material2d for DynamicMaterial {
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Default
    }
    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Blend
    }
    fn specialize(
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        key: Material2dKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        if let Some(fragment) = descriptor.fragment.as_mut() {
            fragment.shader = key.bind_group_data.fragment.clone();
        }
        Ok(())
    }
}

/// On shader (re)load, re-parse + cache the schema. **Only fires for
/// the shader the DynamicBackground entity currently uses** — Bevy's
/// AssetEvent stream fires for every Shader asset (including the dozen
/// or so internal mesh2d / view / utility shaders Bevy ships), and
/// trying to introspect those is both wrong and very noisy.
pub fn refresh_shader_schemas(
    mut events: MessageReader<AssetEvent<Shader>>,
    shaders: Res<Assets<Shader>>,
    schemas: Res<ShaderSchemas>,
    backgrounds: Query<&DynamicBackground>,
) {
    // Build the allowlist: only IDs we actually care about.
    let allowed: std::collections::HashSet<AssetId<Shader>> =
        backgrounds.iter().map(|b| b.shader.id()).collect();
    for ev in events.read() {
        let (AssetEvent::Added { id }
        | AssetEvent::Modified { id }
        | AssetEvent::LoadedWithDependencies { id }) = ev
        else {
            continue;
        };
        if !allowed.contains(id) {
            continue;
        }
        let Some(shader) = shaders.get(*id) else { continue };
        let source = match shader_source_str(shader) {
            Some(s) => s,
            None => continue,
        };
        match Schema::from_wgsl(&source) {
            Ok(schema) => {
                eprintln!(
                    "[dynamic] schema refreshed: {} fields, {} textures, {} samplers",
                    schema.fields.len(),
                    schema.textures.len(),
                    schema.samplers.len(),
                );
                schemas.set(*id, schema);
            }
            Err(e) => {
                eprintln!("[dynamic] schema parse failed: {}", e);
            }
        }
    }
}

fn shader_source_str(shader: &Shader) -> Option<String> {
    // Bevy 0.18's Shader stores its source in a public `source: Source`
    // field. `Source::as_str()` returns the underlying WGSL/Wesl/Glsl
    // text; for our purposes we only care about WGSL.
    Some(shader.source.as_str().to_string())
}

// ============================================================
// DynamicMaterialPlugin — spawns the canvas quad, applies queued
// script operations to the material, and refreshes the shared
// snapshot scripts read from.
// ============================================================

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
use bevy::camera::ClearColorConfig;
use bevy::image::Image;
use bevy::mesh::Mesh2d;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::sprite_render::{Material2dPlugin, MeshMaterial2d};

use crate::material::STYLE_SOURCE;
use crate::script_bridge::{
    drain_script_msgs, fire_scheduled_events, EventBus, MaskOp, PaneRectSnap, PendingScriptOps,
    ScriptSnapshot, UniformWrite,
};
use crate::shader::ActiveProject;

/// Render layer reserved for the dynamic canvas overlay.
const OVERLAY_LAYER: usize = 30;

/// Path inside the `style://` asset source where the user's canvas
/// shader lives. The host bootstraps this from an embedded default
/// on first launch; otherwise leaves user edits alone.
const SHADER_ASSET_PATH: &str = ".editor/shaders/dust.wgsl";

/// Per-project mask textures, keyed by `(project_id, mask_name)`.
/// Lazy-allocated on first `mask_paint`/`mask_fill` for a given name.
/// Sized to a fixed [`MASK_SIZE`] square and UV-mapped across the
/// whole canvas regardless of window aspect.
pub const MASK_SIZE: u32 = 1024;

#[derive(Resource, Default)]
pub struct DynamicMasks {
    pub by_key: HashMap<(u64, String), Handle<Image>>,
}

/// Per-shader cached schema's handle of the shader it belongs to.
/// Held on the background entity so systems know which schema applies
/// when applying script writes.
#[derive(Component)]
pub struct DynamicBackground {
    pub shader: Handle<Shader>,
}

#[derive(Resource, Default)]
pub struct DynamicBackgroundEntity(pub Option<Entity>);

/// Embedded fallback shader text. Written to disk on first launch and
/// then never touched — disk file is the source of truth from then on.
const EMBEDDED_DUST_SHADER: &str = include_str!("dust_default.wgsl");

pub struct DynamicMaterialPlugin;

impl Plugin for DynamicMaterialPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(Material2dPlugin::<DynamicMaterial>::default())
            .init_resource::<ShaderSchemas>()
            .init_resource::<DynamicMasks>()
            .init_resource::<DynamicBackgroundEntity>()
            .init_resource::<PendingScriptOps>()
            .add_systems(
                Startup,
                (
                    ensure_bootstrap_dust_shader,
                    spawn_overlay_camera,
                )
                    .chain(),
            )
            .add_systems(
                Update,
                (
                    spawn_dynamic_background,
                    refresh_shader_schemas,
                    drain_script_msgs,
                    fire_scheduled_events,
                    emit_engine_events,
                    apply_uniform_writes,
                    apply_mask_ops,
                    resize_dynamic_quad,
                    refresh_snapshot,
                )
                    .chain(),
            );
    }
}

fn ensure_bootstrap_dust_shader(data_dir: Option<Res<crate::state::StyleDataDir>>) {
    let Some(data_dir) = data_dir else { return };
    let shaders_dir = data_dir.0.join(".editor").join("shaders");
    let _ = std::fs::create_dir_all(&shaders_dir);
    let dust_path = shaders_dir.join("dust.wgsl");
    if !dust_path.exists() {
        if let Err(e) = std::fs::write(&dust_path, EMBEDDED_DUST_SHADER) {
            eprintln!("[dynamic] bootstrap write {:?}: {}", dust_path, e);
        } else {
            eprintln!("[dynamic] wrote bootstrap shader {:?}", dust_path);
        }
    }
}

fn spawn_overlay_camera(mut commands: Commands) {
    commands.spawn((
        Camera2d,
        Camera {
            // Above every pane camera + sidebar. The dust overlay
            // covers the whole canvas including panes.
            order: 1_000_001,
            clear_color: ClearColorConfig::None,
            ..default()
        },
        RenderLayers::layer(OVERLAY_LAYER),
        Name::new("dynamic::overlay_camera"),
    ));
}

fn spawn_dynamic_background(
    mut commands: Commands,
    mut entity: ResMut<DynamicBackgroundEntity>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<DynamicMaterial>>,
    mut images: ResMut<Assets<Image>>,
    asset_server: Res<AssetServer>,
) {
    if entity.0.is_some() {
        return;
    }
    let blank = images.add(blank_mask());
    let shader: Handle<Shader> =
        asset_server.load::<Shader>(format!("{}://{}", STYLE_SOURCE, SHADER_ASSET_PATH));
    let mesh = meshes.add(Rectangle::new(1.0, 1.0));
    let material = materials.add(DynamicMaterial {
        user: UserBuffer::default(),
        tex0: blank.clone(),
        tex1: blank.clone(),
        tex2: blank.clone(),
        tex3: blank.clone(),
        tex4: blank.clone(),
        tex5: blank.clone(),
        tex6: blank.clone(),
        tex7: blank,
        fragment: shader.clone(),
    });
    let e = commands
        .spawn((
            Mesh2d(mesh),
            MeshMaterial2d(material),
            Transform::from_xyz(0.0, 0.0, 0.0),
            RenderLayers::layer(OVERLAY_LAYER),
            DynamicBackground { shader },
            Name::new("dynamic::background"),
        ))
        .id();
    entity.0 = Some(e);
}

fn resize_dynamic_quad(
    bg: Res<DynamicBackgroundEntity>,
    mut transforms: Query<&mut Transform>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
) {
    let Some(e) = bg.0 else { return };
    let Ok(window) = windows.single() else { return };
    if let Ok(mut t) = transforms.get_mut(e) {
        t.scale.x = window.width().max(1.0);
        t.scale.y = window.height().max(1.0);
    }
}

fn apply_uniform_writes(
    bg: Res<DynamicBackgroundEntity>,
    backgrounds: Query<&DynamicBackground>,
    handles: Query<&MeshMaterial2d<DynamicMaterial>>,
    mut materials: ResMut<Assets<DynamicMaterial>>,
    schemas: Res<ShaderSchemas>,
    mut pending: ResMut<PendingScriptOps>,
) {
    let Some(e) = bg.0 else { return };
    let Ok(handle) = handles.get(e) else { return };
    let Ok(bg_comp) = backgrounds.get(e) else { return };
    let Some(material) = materials.get_mut(&handle.0) else { return };
    let Some(schema) = schemas.get(bg_comp.shader.id()) else {
        // Schema not loaded yet — drop writes for this frame; script
        // will re-issue them next tick. (Avoids racing the shader's
        // first parse.)
        pending.uniform_writes.clear();
        return;
    };

    let buf = material.user.as_bytes_mut();
    for write in pending.uniform_writes.drain(..) {
        match write {
            UniformWrite::F32(name, v) => schema.write_f32(buf, &name, v),
            UniformWrite::Vec2(name, v) => schema.write_vec2(buf, &name, v),
            UniformWrite::Vec4(name, v) => schema.write_vec4(buf, &name, v),
        }
    }
}

fn apply_mask_ops(
    bg: Res<DynamicBackgroundEntity>,
    backgrounds: Query<&DynamicBackground>,
    handles: Query<&MeshMaterial2d<DynamicMaterial>>,
    mut materials: ResMut<Assets<DynamicMaterial>>,
    schemas: Res<ShaderSchemas>,
    mut masks: ResMut<DynamicMasks>,
    mut images: ResMut<Assets<Image>>,
    active: Res<crate::shader::ActiveProject>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    mut pending: ResMut<PendingScriptOps>,
) {
    let Some(e) = bg.0 else { return };
    let Ok(handle) = handles.get(e) else { return };
    let Ok(bg_comp) = backgrounds.get(e) else { return };
    let Some(material) = materials.get_mut(&handle.0) else { return };
    let Some(schema) = schemas.get(bg_comp.shader.id()) else {
        pending.mask_ops.clear();
        return;
    };
    let Ok(window) = windows.single() else { return };
    let win_size = Vec2::new(window.width().max(1.0), window.height().max(1.0));
    let project_id = active.0.unwrap_or(0);

    for op in pending.mask_ops.drain(..) {
        match op {
            MaskOp::Paint { name, x, y, radius, value } => {
                // Get-or-create the image for this (project, name).
                let key = (project_id, name.clone());
                let mask_handle = masks
                    .by_key
                    .entry(key)
                    .or_insert_with(|| images.add(blank_mask()))
                    .clone();
                // Bind into the material at the binding the shader
                // declared for this texture name.
                if let Some(binding) = schema.textures.get(&name) {
                    material.set_texture(*binding, mask_handle.clone());
                }
                if let Some(image) = images.get_mut(&mask_handle) {
                    paint_brush(image, x, y, radius, value, win_size);
                }
            }
            MaskOp::Fill(name, value) => {
                let key = (project_id, name.clone());
                let mask_handle = masks
                    .by_key
                    .entry(key)
                    .or_insert_with(|| images.add(blank_mask()))
                    .clone();
                if let Some(binding) = schema.textures.get(&name) {
                    material.set_texture(*binding, mask_handle.clone());
                }
                if let Some(image) = images.get_mut(&mask_handle)
                    && let Some(data) = image.data.as_mut()
                {
                    let v = (value.clamp(0.0, 1.0) * 255.0) as u8;
                    for chunk in data.chunks_mut(4) {
                        chunk[0] = v;
                    }
                }
            }
        }
    }
}

fn refresh_snapshot(
    time: Res<Time>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    snapshot: Option<Res<ScriptSnapshot>>,
    panes: Query<(&pane_bevy::PaneRect, &pane_bevy::PaneKindMarker), With<pane_bevy::PaneTag>>,
    focused: Res<pane_bevy::FocusedPane>,
    focused_rect: Query<&pane_bevy::PaneRect>,
    mut pending: ResMut<PendingScriptOps>,
) {
    let Some(snapshot) = snapshot else { return };
    let mut pane_rects = Vec::new();
    for (rect, kind) in &panes {
        pane_rects.push(PaneRectSnap {
            x: rect.pos.x,
            y: rect.pos.y,
            w: rect.size.x,
            h: rect.size.y,
            kind: kind.0.to_string(),
        });
    }
    let mut uniforms = serde_json::Map::new();
    uniforms.insert("time".into(), json_num(time.elapsed_secs() as f64));
    uniforms.insert("dt".into(), json_num(time.delta_secs() as f64));
    if let Ok(window) = windows.single() {
        uniforms.insert(
            "resolution".into(),
            serde_json::json!([window.width() as f64, window.height() as f64]),
        );
        if let Some(c) = window.cursor_position() {
            uniforms.insert("mouse_world".into(), serde_json::json!([c.x as f64, c.y as f64]));
        }
    }
    if let Some(e) = focused.0
        && let Ok(rect) = focused_rect.get(e)
    {
        uniforms.insert(
            "focused_pane".into(),
            serde_json::json!([
                rect.pos.x as f64,
                rect.pos.y as f64,
                rect.size.x as f64,
                rect.size.y as f64,
            ]),
        );
    }
    let state_writes: Vec<_> = pending.state_writes.drain(..).collect();
    snapshot.write(|data| {
        data.pane_rects = pane_rects;
        data.uniforms = uniforms;
        // Drain accumulated state_set calls into the shared snapshot.
        // Other scripts see the new values from their next tick.
        for (k, v) in state_writes {
            data.state.insert(k, v);
        }
    });
}

fn json_num(v: f64) -> serde_json::Value {
    serde_json::Number::from_f64(v)
        .map(serde_json::Value::Number)
        .unwrap_or(serde_json::Value::Null)
}

/// Engine → script event bridge. Scripts that want to react to focus
/// changes / project changes / etc. listen via their `events` array
/// for these `kind`s:
/// - `focus_changed` → `{ entity: u64 | null }`
/// - `project_changed` → `{ from: u64 | null, to: u64 | null }`
///
/// Each event is fired exactly once per state change. Uses `Local`s
/// for delta detection so we don't depend on Bevy change-tickers
/// (which would fire on every frame for resources mutated in-place).
fn emit_engine_events(
    focused: Res<pane_bevy::FocusedPane>,
    active: Res<ActiveProject>,
    mut bus: ResMut<EventBus>,
    mut last_focus: Local<Option<Entity>>,
    mut last_project: Local<Option<u64>>,
    mut warmed: Local<bool>,
) {
    if !*warmed {
        *warmed = true;
        *last_focus = focused.0;
        *last_project = active.0;
        return;
    }
    if focused.0 != *last_focus {
        *last_focus = focused.0;
        let payload = match focused.0 {
            Some(e) => serde_json::json!({ "entity": e.to_bits() }),
            None => serde_json::json!({ "entity": null }),
        };
        bus.push("focus_changed", payload);
    }
    if active.0 != *last_project {
        let payload = serde_json::json!({
            "from": *last_project,
            "to": active.0,
        });
        *last_project = active.0;
        bus.push("project_changed", payload);
    }
}

/// Stamp a soft-cosine brush at (x, y) in window-logical pixels into
/// the mask image. The mask is UV-mapped across the window, so we
/// convert window pixels → mask pixels via the window:mask ratio.
fn paint_brush(image: &mut Image, x: f32, y: f32, radius: f32, value: f32, win_size: Vec2) {
    let uv = Vec2::new(x / win_size.x, y / win_size.y);
    let center = uv * MASK_SIZE as f32;
    let w = MASK_SIZE as i32;
    let h = MASK_SIZE as i32;
    let r = radius.ceil() as i32;
    let cx = center.x.round() as i32;
    let cy = center.y.round() as i32;
    let x_lo = (cx - r).max(0);
    let x_hi = (cx + r).min(w - 1);
    let y_lo = (cy - r).max(0);
    let y_hi = (cy + r).min(h - 1);
    let Some(data) = image.data.as_mut() else { return };
    let target = (value.clamp(0.0, 1.0) * 255.0) as u8;
    for y in y_lo..=y_hi {
        let row = (y * w) as usize * 4;
        for x in x_lo..=x_hi {
            let dx = x as f32 - center.x;
            let dy = y as f32 - center.y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist > radius {
                continue;
            }
            let t = (dist / radius).clamp(0.0, 1.0);
            let falloff = (1.0 - t * t).max(0.0);
            let stroke = (falloff * target as f32) as u8;
            let idx = row + (x as usize) * 4;
            if data[idx] < stroke {
                data[idx] = stroke;
            }
        }
    }
}

fn blank_mask() -> Image {
    let bytes = vec![0u8; (MASK_SIZE * MASK_SIZE * 4) as usize];
    Image::new(
        Extent3d {
            width: MASK_SIZE,
            height: MASK_SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        bytes,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    )
}
