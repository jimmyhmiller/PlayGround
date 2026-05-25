//! Pane chrome material — a single `Material2d` per pane body that
//! renders a rounded-rect SDF with optional border and focus glow.
//!
//! Replaces the flat `Sprite` we used to use for the pane body. Same
//! one-quad-per-pane cost, but corners can be rounded, edges
//! anti-aliased, and the visible look is fully data-driven from the
//! material's uniforms (which a theme can rewrite at runtime).
//!
//! See `chrome_material.wgsl` for the shader.

use bevy::asset::{embedded_path, AssetPath};
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dKey, Material2dPlugin};

pub struct ChromeMaterialPlugin;

impl Plugin for ChromeMaterialPlugin {
    fn build(&self, app: &mut App) {
        bevy::asset::embedded_asset!(app, "chrome_material.wgsl");
        bevy::asset::embedded_asset!(app, "shadow_material.wgsl");
        app.init_resource::<ChromeStyle>()
            .init_resource::<ActiveChromeShader>()
            .add_systems(Startup, init_default_chrome_shader)
            .add_plugins(Material2dPlugin::<PaneChromeMaterial>::default())
            .add_plugins(Material2dPlugin::<PaneShadowMaterial>::default());
    }
}

/// Currently-active chrome fragment shader. Defaults to the embedded
/// SDF rounded rect; `style-bevy::presets` overwrites it when the
/// active preset ships its own `chrome.wgsl`. The sync system reacts
/// to `is_changed` and reassigns the fragment handle on every pane's
/// material the same frame.
#[derive(Resource, Default, Clone)]
pub struct ActiveChromeShader(pub Handle<Shader>);

fn init_default_chrome_shader(
    asset_server: Res<AssetServer>,
    mut active: ResMut<ActiveChromeShader>,
) {
    active.0 = asset_server
        .load::<Shader>("embedded://pane_bevy/chrome_material.wgsl");
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct PaneShadowMaterial {
    #[uniform(0)]
    pub params: ShadowParams,
}

impl Material2d for PaneShadowMaterial {
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Path(
            AssetPath::from_path_buf(embedded_path!("shadow_material.wgsl"))
                .with_source("embedded"),
        )
    }
    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Blend
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, ShaderType)]
pub struct ShadowParams {
    /// Mesh extent in pixels (pane size + 2 × blur on each axis).
    pub mesh_size: Vec2,
    /// The actual pane rect inside the mesh.
    pub rect_size: Vec2,
    pub corner_radius: f32,
    pub blur: f32,
    /// Shadow color (rgb + base alpha at the rect edge).
    pub color: Vec4,
    /// Push the shadow down by this many pixels (positive = below).
    pub offset_y: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

/// Theme-driven pane chrome look. Default values match the look the
/// editor shipped with before theming was wired up. A higher-level
/// crate (e.g. `style-bevy`) overrides this from the active theme on
/// `ThemeChanged`; pane-bevy itself never reads the theme directly to
/// avoid a circular dependency.
#[derive(Resource, Clone, Debug)]
pub struct ChromeStyle {
    pub bg: Vec4,
    pub border: Vec4,
    pub border_focused: Vec4,
    /// `(r, g, b, _)` — alpha is ignored, replaced by `focus_strength`.
    pub focus_glow: Vec4,
    pub corner_radius: f32,
    pub border_width: f32,
    pub border_width_focused: f32,
    pub focus_width: f32,
    pub focus_strength: f32,
    // --- drop shadow (separate quad on layer 0, extends outside pane) ---
    /// Shadow color (rgb + base alpha at the rect edge).
    pub shadow_color: Vec4,
    /// How far the shadow fades, pixels. Doubles as the mesh padding
    /// around the pane on every side.
    pub shadow_blur: f32,
    /// Push the shadow down so it sits below the pane. Positive only.
    pub shadow_offset_y: f32,
}

impl Default for ChromeStyle {
    fn default() -> Self {
        let bg = Color::srgb(0.105, 0.110, 0.122).to_linear();
        let border = Color::srgb(0.18, 0.19, 0.22).to_linear();
        let border_focused = Color::srgb(0.30, 0.40, 0.55).to_linear();
        let focus_glow = Color::srgb(0.42, 0.62, 0.92).to_linear();
        let shadow_color = Color::srgba(0.0, 0.0, 0.0, 0.45).to_linear();
        Self {
            bg: Vec4::new(bg.red, bg.green, bg.blue, bg.alpha),
            border: Vec4::new(border.red, border.green, border.blue, border.alpha),
            border_focused: Vec4::new(
                border_focused.red,
                border_focused.green,
                border_focused.blue,
                border_focused.alpha,
            ),
            focus_glow: Vec4::new(focus_glow.red, focus_glow.green, focus_glow.blue, 1.0),
            corner_radius: 6.0,
            border_width: 1.0,
            border_width_focused: 1.5,
            focus_width: 8.0,
            focus_strength: 0.35,
            shadow_color: Vec4::new(
                shadow_color.red,
                shadow_color.green,
                shadow_color.blue,
                shadow_color.alpha,
            ),
            shadow_blur: 24.0,
            shadow_offset_y: 6.0,
        }
    }
}

impl ChromeStyle {
    /// Build `ShadowParams` for a pane of the given size. Mesh extent
    /// pads the rect by `shadow_blur` on each side so the falloff can
    /// reach 0 inside the mesh.
    pub fn shadow_params_for(&self, size: Vec2) -> ShadowParams {
        let mesh_size = Vec2::new(
            size.x + 2.0 * self.shadow_blur,
            size.y + 2.0 * self.shadow_blur,
        );
        ShadowParams {
            mesh_size,
            rect_size: size,
            corner_radius: self.corner_radius,
            blur: self.shadow_blur,
            color: self.shadow_color,
            offset_y: self.shadow_offset_y,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }

    /// Build a `ChromeParams` for a specific pane: combines theme look
    /// (this resource) with per-pane state (size + focus). `time` is
    /// rewritten every frame by `push_chrome_time`, so the initial
    /// value here doesn't matter.
    pub fn params_for(&self, size: Vec2, focused: bool) -> ChromeParams {
        ChromeParams {
            size,
            corner_radius: self.corner_radius,
            border_width: if focused { self.border_width_focused } else { self.border_width },
            bg: self.bg,
            border: if focused { self.border_focused } else { self.border },
            focus: Vec4::new(
                self.focus_glow.x,
                self.focus_glow.y,
                self.focus_glow.z,
                if focused { self.focus_strength } else { 0.0 },
            ),
            focus_width: self.focus_width,
            time: 0.0,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
#[bind_group_data(ChromeMaterialKey)]
pub struct PaneChromeMaterial {
    #[uniform(0)]
    pub params: ChromeParams,
    /// Swappable fragment shader. Bevy caches the render pipeline per
    /// unique fragment handle via `ChromeMaterialKey` so swapping
    /// shaders at runtime doesn't blow up the pipeline cache.
    pub fragment: Handle<Shader>,
}

#[derive(Hash, PartialEq, Eq, Clone)]
pub struct ChromeMaterialKey {
    pub fragment: Handle<Shader>,
}

impl From<&PaneChromeMaterial> for ChromeMaterialKey {
    fn from(m: &PaneChromeMaterial) -> Self {
        Self { fragment: m.fragment.clone() }
    }
}

impl Material2d for PaneChromeMaterial {
    fn fragment_shader() -> ShaderRef {
        // `specialize` overrides this with whatever the material is
        // carrying; this path is just a sensible fallback that Bevy
        // can use for the initial pipeline descriptor.
        ShaderRef::Path(
            AssetPath::from_path_buf(embedded_path!("chrome_material.wgsl"))
                .with_source("embedded"),
        )
    }

    fn alpha_mode(&self) -> AlphaMode2d {
        // Corners and edges are partially transparent — the per-pane
        // camera has `ClearColorConfig::None`, so transparent pixels
        // let the main canvas show through.
        AlphaMode2d::Blend
    }

    fn specialize(
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        key: Material2dKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Per-pane chrome shader swap. The material's
        // `fragment: Handle<Shader>` field carries the asset; we feed
        // its id into the pipeline descriptor so this pane renders
        // with whatever WGSL the active preset (or default) provides.
        if let Some(fragment) = descriptor.fragment.as_mut() {
            fragment.shader = key.bind_group_data.fragment.clone();
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, ShaderType)]
pub struct ChromeParams {
    pub size: Vec2,
    pub corner_radius: f32,
    pub border_width: f32,
    pub bg: Vec4,
    pub border: Vec4,
    /// `(r, g, b, strength)` — strength of 0 disables the glow.
    pub focus: Vec4,
    pub focus_width: f32,
    /// Driven by `push_chrome_time` each frame. Used by the shader's
    /// focus-pulse sin.
    pub time: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

// Look is driven by the `ChromeStyle` resource (built from theme
// tokens by the host) via `ChromeStyle::params_for(size, focused)`.
