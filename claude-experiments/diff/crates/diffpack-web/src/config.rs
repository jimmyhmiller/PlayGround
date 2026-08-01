//! Resolved, framework-neutral configuration for HTML-rooted Web builds.

use std::path::{Path, PathBuf};

use diffpack_core::transform::Target;
use diffpack_default_loader::driver_config::BuildConfig;

/// Configuration for a generic HTML-rooted web build (`diffpack build`).
#[derive(Debug, Clone)]
pub struct WebConfig {
    pub build: BuildConfig,
    /// The public base every emitted URL is joined under.
    pub base: String,
    /// The HTML page entries for a MULTI-PAGE build, as ordered
    /// `(name, absolute_html_path)` pairs from `build.rollupOptions.input`.
    /// Empty means the single-`index.html` default; the `build` command falls
    /// back to `<root>/index.html`.
    pub inputs: Vec<(String, PathBuf)>,
    /// `server.proxy` rules the dev server forwards. Empty for a build.
    pub proxy: Vec<crate::dev_proxy::ProxyRule>,
    /// `build.outDir` as the config sets it, resolved against the project root.
    /// `None` means the config does not set it, and the caller applies Vite's
    /// `dist` default. A command-line `--out-dir` always WINS over this: an
    /// explicit argument outranks a config file.
    pub out_dir: Option<PathBuf>,
    /// Resolved configuration files whose changes require profile reconstruction.
    pub configuration_files: Vec<PathBuf>,
}

/// Framework-neutral description of one emitted HTML page and its assets.
/// Compatibility adapters can translate these records into their own manifests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmittedPage {
    pub key: String,
    pub file: String,
    pub css: Vec<String>,
    pub src: Option<String>,
}

/// Derives the build config for an HTML-rooted web application.
///
/// This function performs no framework/config-file discovery. Compatibility
/// layers may start with this neutral record and explicitly adapt it.
pub fn derive_web_config(root: &Path) -> Result<WebConfig, String> {
    let root = root
        .canonicalize()
        .map_err(|error| format!("cannot open project root {}: {error}", root.display()))?;
    let root = root.as_path();
    let conditions = ["module", "browser"]
        .iter()
        .map(|condition| condition.to_string())
        .collect();
    let config = WebConfig {
        build: BuildConfig {
            base: "/".to_string(),
            browser_process_shim: false,
            asset_inline_limit: 0,
            aliases: Vec::new(),
            conditions,
            main_fields: Vec::new(),
            virtual_modules: Vec::new(),
            private_chunk_names: Vec::new(),
            target: Target::Client,
            source_policy: std::sync::Arc::new(
                diffpack_default_loader::source_policy::NoSourceIntegrationPolicy,
            ),
            hmr: false,
            source_maps: false,
            // Even a generic build knows the project root, so root-relative
            // `@use "/src/..."` targets resolve; additionalData stays a
            // Vite-mode opt-in below.
            scss: diffpack_default_loader::sass::ScssOptions {
                additional_data: None,
                root: Some(root.to_path_buf()),
            },
            // Vite parity: default raster imports stay bare-URL strings.
            image_import_shape: diffpack_default_loader::ImageImportShape::Url,
            // A generic build still honors a project's PostCSS config and its
            // Less/Stylus, resolving each tool from the project root.
            css_preprocess: diffpack_default_loader::CssPreprocess {
                root: Some(root.to_path_buf()),
                postcss: diffpack_default_loader::postcss::discover(root).map(std::sync::Arc::new),
            },
            // Vite/esbuild parity, and deliberate: `.js` is plain JavaScript, so
            // JSX in it is a syntax error here exactly as it is under Vite.
            jsx_extensions: diffpack_core::parser::JsxExtensions::JsxAndTsxOnly,
            // The BUILD's JSX lowering settings. Empty for a generic build: with
            // no `vite.config` read, the tsconfig that owns each file (which is
            // honored in every mode) is the only input.
            jsx: diffpack_core::transform::JsxConfig::default(),
            // A generic build bundles everything it can resolve.
            server_external_packages: Vec::new(),
        },
        base: "/".to_string(),
        inputs: Vec::new(),
        proxy: Vec::new(),
        out_dir: None,
        configuration_files: Vec::new(),
    };
    Ok(config)
}

/// The `process.env.NODE_ENV` compile-time define, the switch every package that
/// ships both a development and a production build dispatches on:
///
/// ```js
/// if (process.env.NODE_ENV === 'production') module.exports = require('./cjs/react-dom-client.production.js');
/// else module.exports = require('./cjs/react-dom-client.development.js');
/// ```
///
/// Supplying it as a literal (rather than only as the runtime global shim) lets
/// [`diffpack_core::dead_branch`] delete
/// the branch that cannot run. Without it BOTH builds are reachable and both are
/// bundled: React's development build alone is over a megabyte, and shipping it to
/// production users is a correctness problem, not just a size one.
///
/// A value the app's own Vite config already declares wins — this fills in the
/// default Vite itself supplies, it does not override an explicit choice.
pub fn set_node_env(defines: &mut Vec<(String, String)>, mode: &str) {
    const KEY: &str = "process.env.NODE_ENV";
    let value = format!("\"{mode}\"");
    match defines.iter_mut().find(|(key, _)| key == KEY) {
        Some(existing) => existing.1 = value,
        None => defines.push((KEY.to_string(), value)),
    }
}

/// Switches a
/// [`derive_web_config`] result to development. HMR instrumentation on,
/// `process.env.NODE_ENV` defined as `"development"` (so dependencies select
/// their development builds — React's hook warnings and the Fast Refresh renderer
/// hook), the `import.meta.env` mode flipped to `development` (so `DEV`/`PROD` fold
/// the dev way), and the resolve `production` condition swapped for `development`
/// so packages with a `development`/`production` exports map resolve their dev
/// entry. Kept as one function for the same reason as `set_development_mode`: these
/// travel together, and a caller must never set them inconsistently.
pub fn set_web_development_mode(config: &mut WebConfig) {
    config.build.hmr = true;
    if let Some(policy) = config.build.source_policy.development() {
        config.build.source_policy = policy;
    }
    for condition in config.build.conditions.iter_mut() {
        if condition == "production" {
            *condition = "development".to_string();
        }
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
                format!(
                    "cannot copy {} to {}: {error}",
                    from.display(),
                    to.display()
                )
            })?;
            copied += 1;
        }
    }
    Ok(copied)
}
