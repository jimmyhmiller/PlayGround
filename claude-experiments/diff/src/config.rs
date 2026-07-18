//! Native TanStack Start build configuration.
//!
//! Diffpack is a replacement for Vite/Rolldown, not a host for them. This module
//! derives everything the build needs — entry aliases and per-environment resolve
//! conditions — from TanStack Start's conventions and the project filesystem, with
//! no Node, no Vite, and no Rolldown. Reading a value out of a config file is our
//! own parse, not a dependency on the tool that also happens to read it.

use std::path::{Path, PathBuf};

use crate::bundler::BuildConfig;

/// The build environments TanStack Start defines. `nitro` is the production
/// server runtime; `ssr` renders; `client` is the browser build.
pub const ENVIRONMENTS: [&str; 3] = ["client", "ssr", "nitro"];

/// Resolved configuration for one environment.
#[derive(Debug, Clone)]
pub struct AppConfig {
    pub environment: String,
    pub build: BuildConfig,
    /// The module that begins this environment's graph (client/server entry).
    pub entry: Option<PathBuf>,
}

/// Derives the build config for `environment` from convention and the filesystem.
pub fn derive_config(root: &Path, environment: &str) -> Result<AppConfig, String> {
    // Absolute paths throughout: module ids must be absolute so tsconfig `paths`
    // discovery can walk up from each importer.
    let root = root
        .canonicalize()
        .map_err(|error| format!("cannot open project root {}: {error}", root.display()))?;
    let root = root.as_path();
    let src = root.join(src_directory(root));
    let default_entry = root
        .join("node_modules/@tanstack/react-start/dist/plugin/default-entry");

    // A user file overrides the framework default; either must exist.
    let pick = |user: PathBuf, default: PathBuf| -> Option<PathBuf> {
        if user.is_file() {
            Some(user)
        } else if default.is_file() {
            Some(default)
        } else {
            None
        }
    };

    let router_entry = if src.join("router.tsx").is_file() {
        Some(src.join("router.tsx"))
    } else if src.join("router.ts").is_file() {
        Some(src.join("router.ts"))
    } else {
        None
    };
    let start_entry = pick(src.join("start.ts"), default_entry.join("start.ts"));
    let client_entry = pick(src.join("client.tsx"), default_entry.join("client.tsx"));
    let server_entry = pick(src.join("server.ts"), default_entry.join("server.ts"));

    let mut aliases = Vec::new();
    let mut add = |name: &str, path: &Option<PathBuf>| {
        if let Some(path) = path {
            aliases.push((name.to_string(), path.to_string_lossy().into_owned()));
        }
    };
    add("#tanstack-router-entry", &router_entry);
    add("#tanstack-start-entry", &start_entry);
    add("virtual:tanstack-start-client-entry", &client_entry);
    add("virtual:tanstack-start-server-entry", &server_entry);

    // Browser conditions isolate the client from server-only code; the server
    // environments resolve with node conditions.
    let conditions = match environment {
        "client" => ["module", "browser", "production"].as_slice(),
        _ => ["node", "production", "wasm", "unwasm"].as_slice(),
    }
    .iter()
    .map(|condition| condition.to_string())
    .collect();

    let entry = match environment {
        "client" => client_entry,
        _ => server_entry,
    };

    Ok(AppConfig {
        environment: environment.to_string(),
        build: BuildConfig { aliases, conditions },
        entry,
    })
}

/// Reads `srcDirectory` out of `vite.config.ts` if present, else defaults to
/// `src`. This is a plain text read of a value, independent of Vite itself; a
/// native Diffpack config format supersedes it later.
fn src_directory(root: &Path) -> String {
    let Ok(text) = std::fs::read_to_string(root.join("vite.config.ts")) else {
        return "src".to_string();
    };
    let Some(marker) = text.find("srcDirectory:") else {
        return "src".to_string();
    };
    let rest = &text[marker + "srcDirectory:".len()..];
    let Some(open) = rest.find(['\'', '"']) else {
        return "src".to_string();
    };
    let quote = rest.as_bytes()[open] as char;
    let after = &rest[open + 1..];
    match after.find(quote) {
        Some(close) => after[..close].to_string(),
        None => "src".to_string(),
    }
}
