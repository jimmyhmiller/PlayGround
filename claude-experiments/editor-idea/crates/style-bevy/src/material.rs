//! Asset source registration for style-bevy's `style://` URL scheme.
//!
//! Once upon a time this module also held a typed `BackgroundMaterial`
//! with fixed UBO structs for World/Project/Theme; that whole approach
//! was replaced by the dynamic-schema architecture in [`crate::dynamic`].
//! What survives here: the small piece of plumbing that lets the host
//! register an asset source rooted at `<base>/projects/` so anything
//! Bevy loads under `style://...` resolves to a file there (with the
//! standard file watcher, so edits hot-reload).

use std::path::PathBuf;

use bevy::asset::io::{AssetSourceBuilder, AssetSourceId};
use bevy::prelude::*;

use crate::state::StyleDataDir;

/// Asset source name used by style-bevy. Per-project shaders live at
/// `style://<id>/.editor/shaders/<file>.wgsl`. Registered by
/// [`register_style_asset_source`].
pub const STYLE_SOURCE: &str = "style";

/// Asset source rooted at `~/.jim/styles/`. Per-preset
/// shaders are loaded via `preset://<name>/chrome.wgsl`. Registered
/// by [`register_preset_asset_source`].
pub const PRESET_SOURCE: &str = "preset";

/// Register the `preset://` asset source. Must be called BEFORE
/// `DefaultPlugins` (same constraint as [`register_style_asset_source`]).
pub fn register_preset_asset_source(app: &mut App, base_dir: PathBuf) {
    if !base_dir.exists()
        && let Err(e) = std::fs::create_dir_all(&base_dir)
    {
        eprintln!(
            "[style] failed to create preset base dir {:?}: {} — preset shaders will be unavailable",
            base_dir, e
        );
        return;
    }
    let path_str = match base_dir.to_str() {
        Some(s) => s.to_string(),
        None => {
            eprintln!("[style] preset base dir is not utf-8: {:?}", base_dir);
            return;
        }
    };
    app.register_asset_source(
        AssetSourceId::Name(PRESET_SOURCE.into()),
        AssetSourceBuilder::platform_default(&path_str, None),
    );
}

/// Call this BEFORE adding `DefaultPlugins` (which inserts
/// `AssetPlugin`, after which sources can no longer be registered).
/// Registers the `style://` asset source rooted at `base_dir` with
/// hot-reload watching enabled via Bevy's standard file-source
/// watcher.
///
/// Example: if `base_dir` is `~/.jim/projects/`, then
/// `asset_server.load("style://42/.editor/shaders/dust.wgsl")` reads
/// `~/.jim/projects/42/.editor/shaders/dust.wgsl` and
/// hot-reloads on edit.
pub fn register_style_asset_source(app: &mut App, base_dir: PathBuf) {
    if !base_dir.exists()
        && let Err(e) = std::fs::create_dir_all(&base_dir)
    {
        eprintln!(
            "[style] failed to create style base dir {:?}: {} — per-project shaders will be unavailable",
            base_dir, e
        );
        return;
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
