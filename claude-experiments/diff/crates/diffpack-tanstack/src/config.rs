//! Native TanStack Start build configuration, independent of the root CLI crate.

use std::path::{Path, PathBuf};

use diffpack_core::transform::Target;
use diffpack_default_loader::driver_config::BuildConfig;
use diffpack_default_loader::{CssPreprocess, ImageImportShape, sass::ScssOptions};
use diffpack_vite_compat::{import_meta_env::ImportMetaEnv, import_meta_glob::ImportMetaGlob};

pub const ENVIRONMENTS: [&str; 4] = ["client", "ssr", "nitro", "react-server"];

pub type AppConfig = diffpack_default_loader::driver_config::EnvironmentConfig;

pub fn derive_config(root: &Path, environment: &str) -> Result<AppConfig, String> {
    let root = root
        .canonicalize()
        .map_err(|error| format!("cannot open project root {}: {error}", root.display()))?;
    let src = root.join(
        diffpack_vite_compat::vite_config::config_string(&root, "srcDirectory")
            .unwrap_or_else(|| "src".to_string()),
    );
    let defaults = root.join("node_modules/@tanstack/react-start/dist/plugin/default-entry");
    let pick = |user: PathBuf, fallback: PathBuf| {
        user.is_file()
            .then_some(user)
            .or_else(|| fallback.is_file().then_some(fallback))
    };
    let router = [src.join("router.tsx"), src.join("router.ts")]
        .into_iter()
        .find(|path| path.is_file());
    let start = pick(src.join("start.ts"), defaults.join("start.ts"));
    let client = pick(src.join("client.tsx"), defaults.join("client.tsx"));
    let server = pick(src.join("server.ts"), defaults.join("server.ts"));
    let mut aliases = Vec::new();
    for (name, path) in [
        ("#tanstack-router-entry", &router),
        ("#tanstack-start-entry", &start),
        ("virtual:tanstack-start-client-entry", &client),
        ("virtual:tanstack-start-server-entry", &server),
    ] {
        if let Some(path) = path {
            aliases.push((name.to_string(), path.to_string_lossy().into_owned()));
        }
    }
    let conditions = match environment {
        "client" => vec!["module", "browser", "production"],
        "react-server" => vec!["react-server", "node", "production", "wasm", "unwasm"],
        _ => vec!["node", "production", "wasm", "unwasm"],
    }
    .into_iter()
    .map(str::to_string)
    .collect();
    let target = match environment {
        "client" => Target::Client,
        "react-server" => Target::IsolatedServer,
        _ => Target::Server,
    };
    let entry = match environment {
        "client" => client,
        "react-server" => [src.join("rsc-entry.tsx"), src.join("rsc-entry.ts")]
            .into_iter()
            .find(|path| path.is_file())
            .or_else(|| server.clone()),
        _ => server,
    };
    let resolved = match diffpack_vite_compat::vite_config::resolve(&root, "production") {
        Ok(value) => value,
        Err(error) => {
            eprintln!(
                "warning: could not evaluate vite config ({error}); continuing with defaults"
            );
            None
        }
    };
    let base = resolved
        .as_ref()
        .and_then(|value| value.base.clone())
        .unwrap_or_else(|| "/".to_string());
    aliases.extend(
        resolved
            .as_ref()
            .map(|v| v.alias.clone())
            .unwrap_or_default(),
    );
    let mut defines = resolved
        .as_ref()
        .map(|v| v.define.clone())
        .unwrap_or_default();
    set_node_env(&mut defines, "production");
    let vite_vars = std::env::vars()
        .filter(|(name, _)| name.starts_with("VITE_"))
        .collect();
    Ok(AppConfig {
        environment: environment.to_string(),
        entry,
        build: BuildConfig {
            base: base.clone(),
            browser_process_shim: true,
            asset_inline_limit: 4096,
            aliases,
            conditions,
            main_fields: Vec::new(),
            virtual_modules: Vec::new(),
            private_chunk_names: vec![(
                crate::manifest::START_MANIFEST_SPECIFIER.to_string(),
                "_tanstack-start-manifest_v{ext}".to_string(),
            )],
            target,
            source_policy: std::sync::Arc::new(
                diffpack_vite_compat::source_policy::ViteSourcePolicy {
                    import_meta_env: Some(ImportMetaEnv {
                        base,
                        mode: "production".to_string(),
                        vite_vars,
                    }),
                    import_meta_glob: Some(ImportMetaGlob { root: root.clone() }),
                    defines,
                },
            ),
            hmr: false,
            source_maps: false,
            scss: ScssOptions {
                additional_data: resolved
                    .as_ref()
                    .and_then(|v| v.scss_additional_data.clone()),
                root: Some(root.clone()),
            },
            image_import_shape: ImageImportShape::Url,
            css_preprocess: CssPreprocess {
                root: Some(root.clone()),
                postcss: diffpack_default_loader::postcss::discover(&root).map(std::sync::Arc::new),
            },
            jsx_extensions: diffpack_core::parser::JsxExtensions::JsxAndTsxOnly,
            jsx: resolved.map(|v| v.jsx).unwrap_or_default(),
            server_external_packages: Vec::new(),
        },
    })
}

pub fn set_development_mode(config: &mut AppConfig) {
    config.build.hmr = true;
    if let Some(policy) = config.build.source_policy.development() {
        config.build.source_policy = policy;
    }
}

fn set_node_env(defines: &mut Vec<(String, String)>, mode: &str) {
    const KEY: &str = "process.env.NODE_ENV";
    let value = format!("\"{mode}\"");
    match defines.iter_mut().find(|(key, _)| key == KEY) {
        Some(existing) => existing.1 = value,
        None => defines.push((KEY.to_string(), value)),
    }
}
