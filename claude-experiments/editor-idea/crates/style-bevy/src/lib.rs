//! Per-project styling: themes (Rhai-loaded design tokens) + canvas
//! background shaders (live-editable WGSL with engine-provided data).
//!
//! # Two halves, one plugin
//!
//! The plugin is split because the two halves are independently useful:
//! a host that just wants design tokens (colors, sizes) can ignore the
//! shader half; a host that just wants live shaders can ignore the theme
//! half. They wire together at one point: the shader system exposes the
//! theme as a uniform block so shaders can read the project's palette.
//!
//! ## Files on disk
//!
//! Per-project assets live under a host-provided base directory. The
//! host inserts [`StyleDataDir`] at startup pointing at e.g.
//! `~/.terminal-bevy/projects/`. For project id `42`, style-bevy reads:
//!
//! ```text
//! <base>/42/
//!   theme.rhai                  — design tokens (optional)
//!   shaders/background.wgsl     — canvas background shader (optional)
//!   state.json                  — engine-managed temporal state
//! ```
//!
//! Missing files fall back to embedded defaults; errors are surfaced via
//! [`StyleErrors`] but never crash the host.
//!
//! ## Shader data flow
//!
//! Engine data → providers → packed UBO bytes → shader. Each frame:
//!
//! 1. Tier-0/1 (world) providers run once, write to the world UBO.
//! 2. Tier-2 (per-project) providers run once per active project, write
//!    to that project's UBO.
//! 3. The theme is copied into the theme UBO.
//! 4. The background material's bind group references all three.
//!
//! Shaders import the auto-prelude (embedded in this crate) which
//! declares `world`, `proj`, `theme` uniform blocks with matching layout.

use bevy::prelude::*;

pub mod dev;
pub mod material;
pub mod shader;
pub mod state;
pub mod theme;
pub mod wipe;

pub use dev::{dev_sender, register_dev_rhai_fns, DevMsg, DevOverrides};
pub use material::{register_style_asset_source, BackgroundMaterial, BackgroundShaderPlugin};
pub use wipe::{WipeMasks, WipePlugin};
pub use shader::{
    FieldScope, FieldType, FieldValue, ShaderDataRegistry, ShaderField, ShaderFieldsAppExt,
};
pub use state::{ProjectStyleState, StyleDataDir};
pub use theme::{tokens, Theme, ThemeChanged, TokenId, TokenValue};

/// Errors from theme parsing or shader compilation, surfaced as a
/// resource so a status pane can show them. Never causes the host to
/// crash — broken files just fall back to the last good version or to
/// the embedded default.
#[derive(Resource, Default, Debug, Clone)]
pub struct StyleErrors {
    pub theme_error: Option<String>,
    pub shader_error: Option<String>,
}

/// Top-level plugin. Pulls in both halves of the system plus the shader
/// material + canvas background mesh.
pub struct StylePlugin;

impl Plugin for StylePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Theme>()
            .init_resource::<ShaderDataRegistry>()
            .init_resource::<ProjectStyleState>()
            .init_resource::<StyleErrors>()
            .add_message::<ThemeChanged>()
            .add_plugins(dev::DevPlugin)
            .add_plugins(theme::ThemePlugin)
            .add_plugins(shader::ShaderProviderPlugin)
            .add_plugins(wipe::WipePlugin)
            .add_plugins(BackgroundShaderPlugin);
    }
}
