//! Canvas background material + render plumbing.
//!
//! One full-screen quad on render-layer 0 behind everything else. Its
//! material carries the three uniform blocks (world / project / theme)
//! that every shader sees, plus a `Handle<Shader>` whose pointed-at
//! WGSL is what actually runs.
//!
//! Per-project shader swapping is done via the **`bind_group_data`**
//! trick: `Handle<Shader>` lives in the specialization key, so each
//! unique shader handle gets its own pipeline in the cache, and
//! `Material2d::specialize` patches the fragment shader on the
//! descriptor from the key. No custom render command needed.
//!
//! Hot-reload: project shaders are loaded through a dedicated
//! `"style"` asset source rooted at the host-provided base dir. Editing
//! `<base>/<project_id>/.editor/shaders/background.wgsl` triggers
//! AssetServer's normal watcher and re-specializes the pipeline.

use std::path::PathBuf;

use bevy::asset::io::{AssetSourceBuilder, AssetSourceId};
use bevy::camera::visibility::RenderLayers;
use bevy::camera::ClearColorConfig;
use bevy::mesh::{Mesh2d, MeshVertexBufferLayoutRef};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dKey, Material2dPlugin, MeshMaterial2d};

use crate::shader::{ActiveProject, ProjectUniforms, ShaderDataRegistry, ThemeUniforms, WorldUniforms};
use crate::state::StyleDataDir;
use crate::theme::{tokens, Theme, TokenValue};
use crate::wipe::WipeMasks;
use crate::StyleErrors;

/// Asset source name used by style-bevy. Per-project shaders live at
/// `style://<id>/.editor/shaders/background.wgsl`. Registered by
/// [`register_style_asset_source`].
pub const STYLE_SOURCE: &str = "style";

/// Render layer reserved for the dust overlay quad. Pane-bevy's
/// [`PaneLayerAllocator`] hands out small ids starting at 1, so 31
/// stays clear; if that ever changes we'd want to coordinate with the
/// allocator (or expose a reservation API).
const DUST_OVERLAY_LAYER: usize = 31;

/// Material for the canvas-background quad. Holds the three UBOs that
/// every shader reads, plus a `Handle<Shader>` that drives which WGSL
/// is actually executed. `fragment` is **not** a binding — it's
/// extracted into the spec key via [`BackgroundKey`] so per-project
/// shader paths get separate pipeline cache entries.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
#[bind_group_data(BackgroundKey)]
pub struct BackgroundMaterial {
    #[uniform(0)]
    pub world: WorldUniforms,
    #[uniform(1)]
    pub project: ProjectUniforms,
    #[uniform(2)]
    pub theme: ThemeUniforms,
    /// Per-project wipe mask. R channel = how much dust has been
    /// rubbed off at this UV. Shader samples it and uses `1 - wipe`
    /// to attenuate dust alpha. Swapped on project change.
    #[texture(3)]
    #[sampler(4)]
    pub wipe_mask: Handle<Image>,
    /// Active fragment shader. Lives in the material so the host can
    /// swap it when the active project changes.
    pub fragment: Handle<Shader>,
}

#[derive(Hash, PartialEq, Eq, Clone)]
pub struct BackgroundKey {
    pub fragment: Handle<Shader>,
}

impl From<&BackgroundMaterial> for BackgroundKey {
    fn from(m: &BackgroundMaterial) -> Self {
        Self {
            fragment: m.fragment.clone(),
        }
    }
}

impl Material2d for BackgroundMaterial {
    fn fragment_shader() -> ShaderRef {
        // We override per-instance from the spec key in `specialize`;
        // the default here just keeps the pipeline-builder happy.
        ShaderRef::Default
    }

    fn alpha_mode(&self) -> AlphaMode2d {
        // The dust shader outputs translucent grain on top of the
        // canvas (and the panes underneath). Opaque would paint the
        // whole quad solid even where the shader wants alpha=0.
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

/// Handle to the embedded default `default_background.wgsl`. Used as
/// the initial shader and as the fallback when a project's shader file
/// is missing or fails to load.
///
/// `_prelude` is kept around so the prelude.wgsl asset is loaded and
/// its `#define_import_path style_bevy::prelude` registers with the
/// shader-import resolver. Without holding a handle the asset would
/// be dropped and imports would fail.
#[derive(Resource)]
pub struct DefaultBackgroundShader {
    pub shader: Handle<Shader>,
    _prelude: Handle<Shader>,
}

/// Entity holding the full-screen background quad. Cached so we don't
/// have to query for it every frame.
#[derive(Resource, Default)]
pub struct BackgroundEntity(pub Option<Entity>);

pub struct BackgroundShaderPlugin;

impl Plugin for BackgroundShaderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(Material2dPlugin::<BackgroundMaterial>::default())
            .init_resource::<BackgroundEntity>()
            .add_systems(
                Startup,
                // Bootstrap must run before init_default_shader so the
                // disk files exist when AssetServer tries to load them.
                (ensure_bootstrap_shader_files, init_default_shader)
                    .chain()
                    .before(spawn_overlay_camera),
            )
            .add_systems(Startup, spawn_overlay_camera)
            .add_systems(
                Update,
                (
                    spawn_background_quad,
                    swap_project_shader,
                    resize_background_quad,
                    update_dust_visibility,
                    push_uniforms,
                )
                    .chain(),
            );
    }
}

/// The WGSL files baked into the binary. These are the **bootstrap**
/// copies — the engine never loads them at runtime. On first launch
/// the host writes them to disk at `<base>/.editor/...` and reads back
/// through the `style://` asset source, which has a file watcher that
/// turns every save into a hot reload. Edit the on-disk copy to
/// iterate live; only re-run a build if you change one of these
/// embedded strings (or any Rust code).
const EMBEDDED_PRELUDE: &str = include_str!("prelude.wgsl");
const EMBEDDED_DEFAULT_BACKGROUND: &str = include_str!("default_background.wgsl");

/// Path inside the style asset source (relative to the source root)
/// for the shared prelude. Stable so the same handle is used across
/// the whole app.
const PRELUDE_ASSET_PATH: &str = ".editor/prelude.wgsl";
/// Same for the shared default background shader.
const DEFAULT_BG_ASSET_PATH: &str = ".editor/default_background.wgsl";

/// One-shot startup system: write the embedded prelude + default
/// shader to disk if they don't already exist. Never overwrites — user
/// edits to these files survive any number of restarts. Deleting a
/// file restores the embedded version on the next launch.
fn ensure_bootstrap_shader_files(data_dir: Option<Res<StyleDataDir>>) {
    let Some(data_dir) = data_dir else { return };
    let editor_dir = data_dir.0.join(".editor");
    if let Err(e) = std::fs::create_dir_all(&editor_dir) {
        eprintln!(
            "[style] failed to create {:?}: {} — shader files will not exist",
            editor_dir, e
        );
        return;
    }
    write_if_missing(&editor_dir.join("prelude.wgsl"), EMBEDDED_PRELUDE);
    write_if_missing(
        &editor_dir.join("default_background.wgsl"),
        EMBEDDED_DEFAULT_BACKGROUND,
    );
}

fn write_if_missing(path: &std::path::Path, contents: &str) {
    if path.exists() {
        return;
    }
    if let Err(e) = std::fs::write(path, contents) {
        eprintln!("[style] failed to write bootstrap file {:?}: {}", path, e);
    } else {
        eprintln!("[style] wrote bootstrap file {:?}", path);
    }
}

/// A dedicated camera that renders ONLY the dust overlay layer. Order
/// is chosen high enough to come after every pane camera (pane orders
/// are `z * 100 + 1` and sidebar is around 50_000) and after the main
/// camera (order 0). Without this camera the dust quad would only
/// render on whatever layer-0 camera covers the canvas — i.e. behind
/// every pane — so dust never overlaid pane content.
///
/// `ClearColorConfig::None` lets the camera composite over whatever
/// previous cameras drew rather than wiping the framebuffer.
fn spawn_overlay_camera(mut commands: Commands) {
    commands.spawn((
        Camera2d,
        Camera {
            order: 1_000_000,
            clear_color: ClearColorConfig::None,
            ..default()
        },
        RenderLayers::layer(DUST_OVERLAY_LAYER),
        Name::new("style::overlay_camera"),
    ));
}

fn init_default_shader(mut commands: Commands, asset_server: Res<AssetServer>) {
    // Load via the `style://` asset source so AssetServer's file
    // watcher picks up edits to either file and re-uploads the GPU
    // shader live — no relaunch required. The prelude handle is kept
    // alive so the `#define_import_path style_bevy::prelude` line
    // stays registered with the shader-import resolver; user shaders
    // depend on it via `#import style_bevy::prelude::{...}`.
    let prelude: Handle<Shader> =
        asset_server.load::<Shader>(format!("{}://{}", STYLE_SOURCE, PRELUDE_ASSET_PATH));
    let shader: Handle<Shader> =
        asset_server.load::<Shader>(format!("{}://{}", STYLE_SOURCE, DEFAULT_BG_ASSET_PATH));
    commands.insert_resource(DefaultBackgroundShader {
        shader,
        _prelude: prelude,
    });
}

/// Create the background quad once we have the default shader handle.
/// Runs every frame but is a no-op after the first successful spawn.
/// The quad is a unit Rectangle scaled to the window each frame, so
/// the mesh itself is never re-uploaded on resize.
fn spawn_background_quad(
    mut commands: Commands,
    mut entity: ResMut<BackgroundEntity>,
    default_shader: Option<Res<DefaultBackgroundShader>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<BackgroundMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    if entity.0.is_some() {
        return;
    }
    let Some(default_shader) = default_shader else { return };

    let mesh = meshes.add(Rectangle::new(1.0, 1.0));
    // Start with a placeholder blank mask; the per-project mask gets
    // swapped in by `swap_project_shader` once we know the active
    // project.
    let blank_mask = images.add(crate::wipe::blank_mask());
    let material = materials.add(BackgroundMaterial {
        world: WorldUniforms::default(),
        project: ProjectUniforms::default(),
        theme: ThemeUniforms::default(),
        wipe_mask: blank_mask,
        fragment: default_shader.shader.clone(),
    });

    // Dust overlays everything because it lives on its own render
    // layer drawn by a dedicated overlay camera that runs after all
    // pane cameras (see `spawn_overlay_camera`). The pane cameras
    // don't render this layer, so dust never appears inside a pane
    // unless the overlay camera paints it on top.
    //
    // The Z coordinate within the overlay camera's view is meaningless
    // here — the layer + camera-order chain decides the compositing.
    let e = commands
        .spawn((
            Mesh2d(mesh),
            MeshMaterial2d(material),
            Transform::from_xyz(0.0, 0.0, 0.0),
            RenderLayers::layer(DUST_OVERLAY_LAYER),
            Name::new("style::dust_overlay"),
        ))
        .id();
    entity.0 = Some(e);
}

/// Stretch the unit-Rectangle to fill the window via Transform scale.
fn resize_background_quad(
    bg: Res<BackgroundEntity>,
    mut transforms: Query<&mut Transform>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
) {
    let Some(e) = bg.0 else { return };
    let Ok(window) = windows.single() else { return };
    let w = window.width().max(1.0);
    let h = window.height().max(1.0);
    if let Ok(mut t) = transforms.get_mut(e) {
        t.scale.x = w;
        t.scale.y = h;
    }
}

/// When the active project changes, swap the material's `fragment`
/// handle to point at the project's WGSL file (or the embedded
/// default if it doesn't exist / fails to load), and swap in the
/// project's wipe mask.
fn swap_project_shader(
    active: Res<ActiveProject>,
    bg: Res<BackgroundEntity>,
    materials_by_entity: Query<&MeshMaterial2d<BackgroundMaterial>>,
    mut materials: ResMut<Assets<BackgroundMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut wipe_masks: ResMut<WipeMasks>,
    asset_server: Res<AssetServer>,
    default_shader: Option<Res<DefaultBackgroundShader>>,
    data_dir: Option<Res<StyleDataDir>>,
    mut errors: ResMut<StyleErrors>,
    mut last_active: Local<Option<u64>>,
) {
    if !active.is_changed() && last_active.is_some() {
        return;
    }
    if *last_active == active.0 {
        return;
    }
    *last_active = active.0;

    let Some(e) = bg.0 else { return };
    let Ok(mat_handle) = materials_by_entity.get(e) else { return };
    let Some(default_shader) = default_shader else { return };
    let Some(material) = materials.get_mut(&mat_handle.0) else { return };

    let new_handle = match (active.0, data_dir.as_ref()) {
        (Some(pid), Some(dir)) => {
            // Check existence on disk first — AssetServer's load for a
            // missing path returns a handle that fails asynchronously,
            // and the pipeline cache then keeps a broken specialization.
            // Cheaper to short-circuit to the embedded default.
            let file = dir
                .0
                .join(pid.to_string())
                .join(".editor/shaders/background.wgsl");
            if file.exists() {
                let path = format!("{}://{}/.editor/shaders/background.wgsl", STYLE_SOURCE, pid);
                asset_server.load::<Shader>(path)
            } else {
                default_shader.shader.clone()
            }
        }
        _ => default_shader.shader.clone(),
    };

    material.fragment = new_handle;

    // Swap the wipe mask to whichever one belongs to this project.
    // Create one lazily if it's the first time the project's been
    // active in this session.
    if let Some(pid) = active.0 {
        let mask_handle = wipe_masks
            .by_project
            .entry(pid)
            .or_insert_with(|| images.add(crate::wipe::blank_mask()))
            .clone();
        material.wipe_mask = mask_handle;
    }

    errors.shader_error = None;
}

/// Toggle the dust overlay's Bevy `Visibility` based on the active
/// theme's `dust_intensity`. When `0` the entity is hidden and the
/// fragment shader doesn't even run — per-project opt-out at the
/// rendering level, not just a transparent output. Hot-reload friendly:
/// the theme watcher updates `Theme`, this system updates the next
/// frame.
fn update_dust_visibility(
    bg: Res<BackgroundEntity>,
    theme: Res<Theme>,
    mut q: Query<&mut Visibility>,
) {
    let Some(e) = bg.0 else { return };
    let Ok(mut vis) = q.get_mut(e) else { return };
    let intensity = match theme.get(tokens::DUST_INTENSITY) {
        Some(TokenValue::F32(v)) => v,
        _ => 1.0,
    };
    let target = if intensity > 0.0 {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
    if *vis != target {
        *vis = target;
    }
}

/// Copy this frame's computed uniforms from the registry into the
/// background material. Cheap — three small structs.
fn push_uniforms(
    registry: Res<ShaderDataRegistry>,
    bg: Res<BackgroundEntity>,
    handles: Query<&MeshMaterial2d<BackgroundMaterial>>,
    mut materials: ResMut<Assets<BackgroundMaterial>>,
) {
    let Some(e) = bg.0 else { return };
    let Ok(mat_handle) = handles.get(e) else { return };
    let Some(material) = materials.get_mut(&mat_handle.0) else { return };
    material.world = registry.world;
    material.project = registry.project;
    material.theme = registry.theme;
}

// ---------- Asset source registration ----------

/// Call this BEFORE adding `DefaultPlugins` (which inserts `AssetPlugin`,
/// after which sources can no longer be registered). Registers the
/// `style://` asset source rooted at `base_dir` with hot-reload watching
/// enabled via Bevy's standard file-source watcher.
///
/// Example: if `base_dir` is `~/.terminal-bevy/projects/`, then
/// `asset_server.load("style://42/.editor/shaders/background.wgsl")`
/// reads `~/.terminal-bevy/projects/42/.editor/shaders/background.wgsl`
/// and hot-reloads on edit.
pub fn register_style_asset_source(app: &mut App, base_dir: PathBuf) {
    if !base_dir.exists() {
        if let Err(e) = std::fs::create_dir_all(&base_dir) {
            eprintln!(
                "[style] failed to create style base dir {:?}: {} — per-project shaders will be unavailable",
                base_dir, e
            );
            return;
        }
    }

    let path_str = match base_dir.to_str() {
        Some(s) => s.to_string(),
        None => {
            eprintln!("[style] base dir is not utf-8: {:?}", base_dir);
            return;
        }
    };

    app.register_asset_source(
        AssetSourceId::Name(STYLE_SOURCE.into()),
        AssetSourceBuilder::platform_default(&path_str, None),
    );

    // Insert the data dir as a resource so other systems can pick it up.
    app.insert_resource(StyleDataDir(base_dir));
}
