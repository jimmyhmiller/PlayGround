//! Per-project styling — design tokens + canvas shader effects, driven
//! by user-edited files on disk.
//!
//! # Architecture (post-rewrite)
//!
//! Everything visual is **data on disk + scripts**, not compiled-in
//! Rust types. The runtime is a small fixed surface:
//!
//! - [`material::register_style_asset_source`] sets up `style://`
//!   hot-reloadable asset paths rooted at a host-provided base dir.
//! - [`DynamicMaterialPlugin`] spawns the canvas-overlay quad backed
//!   by [`DynamicMaterial`] — an opaque 2 KiB uniform buffer + 8
//!   named texture slots + a `Handle<Shader>` chosen by `bind_group_data`
//!   so per-shader pipelines stay cached.
//! - [`introspect::Schema::from_wgsl`] parses the user's WGSL at load
//!   time (via naga) and learns its `UserData` struct fields' offsets
//!   plus its texture-binding names; the host writes values into the
//!   buffer by **name**, not by type.
//! - [`ScriptBridgePlugin`] registers Rhai host fns (`uniform_set`,
//!   `mask_paint`, `emit`, `state_set`, etc.) and routes worker calls
//!   to the main thread via mpsc + a read-side `ScriptSnapshot`.
//! - [`SystemScriptPlugin`] runs a headless Rhai script each tick
//!   (throttled to 30 Hz). Hot-reloads on disk save. Reads engine
//!   events (`focus_changed`, `project_changed`, scheduled emits)
//!   from the shared `EventBus`.
//!
//! Adding a visual behavior is *purely* on-disk:
//! 1. Declare any fields you want in your shader's `UserData` struct.
//! 2. Write a Rhai script that uses `uniform_set`/`mask_paint`/etc.
//!    by those names.
//! 3. Save. AssetServer + the notify watcher reload both files; no
//!    rebuild.

use bevy::prelude::*;

pub mod active;
pub mod chrome_theme;
pub mod dynamic;
pub mod fonts;
pub mod introspect;
pub mod material;
pub mod oklab;
pub mod presets;
pub mod script_bridge;
pub mod script_host;
pub mod state;
pub mod theme;
pub mod theme_bridge;

pub use active::ActiveProject;
pub use dynamic::{DynamicMaterial, DynamicMaterialPlugin, ShaderSchemas};
pub use fonts::{FontRegistry, FontRegistryPlugin};
pub use material::{register_preset_asset_source, register_style_asset_source};
pub use presets::{
    register_preset_host_fns, ActiveStylePreset, PresetsPlugin, StylePreset,
    StylePresetRegistry,
};
pub use script_bridge::{register_script_host_fns, EventBus, ScriptBridgePlugin};
pub use theme_bridge::{register_theme_host_fns, ThemeBridgePlugin};
pub use script_host::SystemScriptPlugin;
pub use state::{ProjectStyleState, StyleDataDir};
pub use theme::{tokens, Theme, ThemeChanged, TokenId, TokenValue};

// Compatibility re-exports for hosts that still spell paths the old
// way. Once terminal-bevy switches to `style_bevy::ActiveProject`,
// these can go.
pub mod shader {
    pub use crate::active::ActiveProject;
}

/// Errors from theme parsing surfaced as a resource so a status pane
/// can show them. Never crashes the host — broken files just keep
/// the last good version.
#[derive(Resource, Default, Debug, Clone)]
pub struct StyleErrors {
    pub theme_error: Option<String>,
}

/// Top-level plugin. Theme system + dynamic shader runtime + the
/// dust system script as the canonical first effect.
pub struct StylePlugin;

impl Plugin for StylePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Theme>()
            .init_resource::<ProjectStyleState>()
            .init_resource::<StyleErrors>()
            .init_resource::<ActiveProject>()
            .add_message::<ThemeChanged>()
            .add_plugins(theme::ThemePlugin)
            .add_plugins(FontRegistryPlugin)
            .add_plugins(chrome_theme::ChromeThemePlugin)
            .add_plugins(PresetsPlugin)
            .add_plugins(ScriptBridgePlugin)
            .add_plugins(ThemeBridgePlugin)
            .add_plugins(DynamicMaterialPlugin)
            .add_plugins(SystemScriptPlugin {
                path: dust_script_path(),
                bootstrap_source: Some(include_str!("dust_default.rhai")),
            });
    }
}

fn dust_script_path() -> std::path::PathBuf {
    let mut p = std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    p.push(".terminal-bevy");
    p.push("widgets");
    p.push("dust.rhai");
    p
}
