//! Native TanStack Start build configuration.
//!
//! Diffpack is a replacement for Vite/Rolldown, not a host for them. This module
//! derives everything the build needs — entry aliases and per-environment resolve
//! conditions — from TanStack Start's conventions and the project filesystem, with
//! no Node, no Vite, and no Rolldown. Reading a value out of a config file is our
//! own parse, not a dependency on the tool that also happens to read it.

use std::path::{Path, PathBuf};

use crate::bundler::BuildConfig;
use crate::transform::Target;

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

    // The client build specializes TanStack Start's environment-directive
    // helpers so server-only code (and its `node:async_hooks` dependency) is
    // tree-shaken out of the browser bundle; server builds keep the neutral
    // runtime stubs.
    let target = match environment {
        "client" => Target::Client,
        _ => Target::Server,
    };

    // Evaluate the Vite config once (in a short-lived `node`, like Vite itself) for
    // its computed values — `define` entries and `base` — which cannot be read as
    // text. Failure is non-fatal: warn and fall back to conventions, so a build
    // without node, or with a config node cannot evaluate, still proceeds.
    let resolved = match crate::vite_config::resolve(root, "production") {
        Ok(resolved) => resolved,
        Err(error) => {
            eprintln!("warning: could not evaluate vite config ({error}); continuing with defaults");
            None
        }
    };
    let base = resolved
        .as_ref()
        .and_then(|resolved| resolved.base.clone())
        .unwrap_or_else(|| "/".to_string());
    let defines = resolved.map(|resolved| resolved.define).unwrap_or_default();

    Ok(AppConfig {
        environment: environment.to_string(),
        build: BuildConfig {
            aliases,
            conditions,
            virtual_modules: Vec::new(),
            target,
            // A TanStack/Vite app expects Vite's `import.meta.env`; supply it (this
            // is the opt-in — generic bundling leaves `import.meta.env` untouched).
            import_meta_env: Some(import_meta_env(base)),
            defines,
            // Off by default; the dev server flips it on per environment. `build-app`
            // uses this config path with `hmr` false, so production is unaffected.
            hmr: false,
        },
        entry,
    })
}

/// Builds the Vite `import.meta.env` values for a production `build-app`: `MODE`
/// is `production` (so `DEV`/`PROD` fold to `false`/`true`), `BASE_URL` is the
/// resolved config `base`, and every `VITE_*` process-env variable present at
/// build time is captured. `SSR` is derived per-target by the rewrite, not here.
fn import_meta_env(base: String) -> crate::import_meta_env::ImportMetaEnv {
    let vite_vars = std::env::vars()
        .filter(|(name, _)| name.starts_with("VITE_"))
        .collect();
    crate::import_meta_env::ImportMetaEnv {
        base,
        mode: "production".to_string(),
        vite_vars,
    }
}

/// Copies the app's static `public/` directory verbatim into the build's
/// `public/` output (favicons, `site.webmanifest`, ...), the `publicDir`
/// convention. Returns the number of files copied; zero when the app has no
/// `public/` directory. Emitted chunks/assets are not disturbed.
pub fn copy_static_public(root: &Path, output_public: &Path) -> Result<usize, String> {
    let source = root.join("public");
    if !source.is_dir() {
        return Ok(0);
    }
    copy_dir_into(&source, output_public)
}

fn copy_dir_into(source: &Path, destination: &Path) -> Result<usize, String> {
    std::fs::create_dir_all(destination)
        .map_err(|error| format!("cannot create {}: {error}", destination.display()))?;
    let mut copied = 0;
    let entries = std::fs::read_dir(source)
        .map_err(|error| format!("cannot read {}: {error}", source.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| format!("cannot read {}: {error}", source.display()))?;
        let from = entry.path();
        let to = destination.join(entry.file_name());
        let file_type = entry
            .file_type()
            .map_err(|error| format!("cannot stat {}: {error}", from.display()))?;
        if file_type.is_dir() {
            copied += copy_dir_into(&from, &to)?;
        } else {
            std::fs::copy(&from, &to).map_err(|error| {
                format!("cannot copy {} to {}: {error}", from.display(), to.display())
            })?;
            copied += 1;
        }
    }
    Ok(copied)
}

/// Reads `srcDirectory` out of `vite.config.ts` if present, else defaults to
/// `src`. This is a plain text read of a value, independent of Vite itself; a
/// native Diffpack config format supersedes it later.
fn src_directory(root: &Path) -> String {
    vite_config_string(root, "srcDirectory").unwrap_or_else(|| "src".to_string())
}

/// Reads the string value of a `key: '<value>'` (or `"<value>"`) option out of
/// `vite.config.ts`, if present. The single reader for every scalar option
/// Diffpack derives from the Vite config (`srcDirectory`, `routesDirectory`,
/// `routeFileIgnorePattern`, ...), so the parse lives in exactly one place. It is
/// a plain text read of a quoted literal, not a dependency on Vite: it does not
/// evaluate the config, so a value built from an expression is simply not found
/// (the caller falls back to the convention default).
pub fn vite_config_string(root: &Path, key: &str) -> Option<String> {
    let text = std::fs::read_to_string(root.join("vite.config.ts")).ok()?;
    let marker = format!("{key}:");
    let start = text.find(&marker)?;
    let rest = &text[start + marker.len()..];
    // Only accept a quote that starts the value (allowing whitespace after the
    // colon); anything else means the value is not a plain string literal.
    let lead = rest.len() - rest.trim_start().len();
    let rest = &rest[lead..];
    let quote = *rest.as_bytes().first()? as char;
    if quote != '\'' && quote != '"' {
        return None;
    }
    let after = &rest[1..];
    let close = after.find(quote)?;
    Some(after[..close].to_string())
}
