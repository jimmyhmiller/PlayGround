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
            .init_resource::<ChromeTextStyle>()
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

/// Theme-driven colors for the chrome text/sprite children that the
/// SDF material doesn't cover: title text, the close button, the
/// resize-handle square, and the title divider line. Same bridge
/// pattern as `ChromeStyle` — pane-bevy reads it, style-bevy writes
/// it from theme tokens.
#[derive(Resource, Clone, Debug)]
pub struct ChromeTextStyle {
    pub title: Color,
    pub title_focused: Color,
    pub divider: Color,
    pub close: Color,
    pub handle: Color,
}

impl Default for ChromeTextStyle {
    fn default() -> Self {
        Self {
            title: Color::srgb(0.62, 0.65, 0.70),
            title_focused: Color::srgb(0.95, 0.96, 0.98),
            divider: Color::srgb(0.165, 0.170, 0.188),
            close: Color::srgb(0.50, 0.52, 0.56),
            handle: Color::srgb(0.22, 0.23, 0.26),
        }
    }
}

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
/// Per-pane chrome look override. When present on a pane, the chrome
/// systems use this instead of the global [`ChromeStyle`] resource — so
/// each pane can render in its own project's theme. The host stamps it
/// from the per-project theme cache; panes without it fall back to the
/// global (active) style.
#[derive(Component, Clone, Debug)]
pub struct PaneChromeStyle(pub ChromeStyle);

/// Per-pane chrome *shader* override. When present, the pane's chrome
/// material uses this fragment shader instead of the global
/// [`ActiveChromeShader`] — so a project whose preset ships a custom
/// `chrome.wgsl` renders with it even in the cube overview, where panes
/// from many projects (and presets) are on screen at once. Panes without
/// it fall back to the active shader.
#[derive(Component, Clone, Debug)]
pub struct PaneChromeShader(pub Handle<Shader>);

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
    // --- title-bar fill (separate from body bg) ---
    /// Title-bar background, unfocused state.
    pub title_bg: Vec4,
    /// Title-bar background, focused state. Different shade is the
    /// primary visual cue for focus.
    pub title_bg_focused: Vec4,
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
        let title_bg = Color::srgb(0.085, 0.090, 0.105).to_linear();
        let title_bg_focused = Color::srgb(0.135, 0.145, 0.170).to_linear();
        let shadow_color = Color::srgba(0.0, 0.0, 0.0, 0.45).to_linear();
        let v4 = |c: LinearRgba| Vec4::new(c.red, c.green, c.blue, c.alpha);
        Self {
            bg: v4(bg),
            border: v4(border),
            border_focused: v4(border_focused),
            focus_glow: Vec4::new(focus_glow.red, focus_glow.green, focus_glow.blue, 1.0),
            corner_radius: 6.0,
            border_width: 1.0,
            border_width_focused: 1.5,
            focus_width: 8.0,
            focus_strength: 0.35,
            title_bg: v4(title_bg),
            title_bg_focused: v4(title_bg_focused),
            shadow_color: v4(shadow_color),
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
            cover_mode: 0.0,
            title_h: 0.0,
            title_bg: if focused { self.title_bg_focused } else { self.title_bg },
        }
    }

    /// Params for the title-cover quad. The cover is sized to the
    /// full pane (same mesh + params as the body) so its rounded
    /// corners, gradient, border, and focus glow all match the body
    /// pixel-for-pixel; the shader cuts out the content area via
    /// `cover_mode` + `title_h`, leaving the cover painting *only*
    /// the title region — which is exactly what we need to mask
    /// scrolled content out from under the title bar.
    pub fn params_for_title_cover(
        &self,
        pane_size: Vec2,
        focused: bool,
        title_h: f32,
    ) -> ChromeParams {
        let mut p = self.params_for(pane_size, focused);
        p.cover_mode = 1.0;
        p.title_h = title_h;
        p
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
    /// `1.0` if this material is the title-cover quad (rendered above
    /// the content_root to mask scrolled content out of the title
    /// region). Default 0.0 for the regular pane body. The shader
    /// returns transparent in the content area when this is set, so
    /// the cover overpaints the title region only.
    pub cover_mode: f32,
    /// Title-region height in pixels. Used by the shader (when
    /// `cover_mode > 0`) to know where the title region ends and the
    /// content area begins. Set to `pane_bevy::TITLE_H` on the cover;
    /// 0.0 on the regular pane body.
    pub title_h: f32,
    /// Title-bar background fill (linear RGB + alpha). Only consulted
    /// in cover mode. Picking a different shade for the title strip is
    /// how focus is signalled now that the body / border stay stable
    /// across focus state.
    pub title_bg: Vec4,
}

// Look is driven by the `ChromeStyle` resource (built from theme
// tokens by the host) via `ChromeStyle::params_for(size, focused)`.
