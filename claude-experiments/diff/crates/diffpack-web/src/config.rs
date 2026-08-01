//! Configuration for generic HTML-rooted Web and opt-in Vite builds.

use std::path::{Path, PathBuf};

use diffpack_core::transform::Target;
use diffpack_default_loader::driver_config::BuildConfig;

/// Configuration for a generic HTML-rooted web build (`diffpack build`).
#[derive(Debug, Clone)]
pub struct WebConfig {
    pub build: BuildConfig,
    /// The public base every emitted URL is joined under. Always `/` unless
    /// Vite mode resolved a different `base` from the project's config.
    pub base: String,
    /// Whether Vite conventions are enabled for this build.
    pub vite: bool,
    /// The HTML page entries for a MULTI-PAGE build, as ordered
    /// `(name, absolute_html_path)` pairs from `build.rollupOptions.input`.
    /// Empty means the single-`index.html` default; the `build` command falls
    /// back to `<root>/index.html`.
    pub inputs: Vec<(String, PathBuf)>,
    /// Whether to emit the Vite build manifest (`build.manifest`), and its file
    /// name (default `.vite/manifest.json`).
    pub emit_manifest: bool,
    pub manifest_name: String,
    /// `server.proxy` rules the dev server forwards. Empty for a build.
    pub proxy: Vec<diffpack_vite_compat::vite_config::ProxyRule>,
    /// `optimizeDeps.exclude` — surfaced for honest reporting (Diffpack does not
    /// pre-bundle, so exclusion is satisfied by construction).
    pub optimize_deps_exclude: Vec<String>,
    /// `build.outDir` as the config sets it, resolved against the project root.
    /// `None` means the config does not set it, and the caller applies Vite's
    /// `dist` default. A command-line `--out-dir` always WINS over this: an
    /// explicit argument outranks a config file.
    pub out_dir: Option<PathBuf>,
}

/// Derives the build config for an HTML-rooted web application.
///
/// The default is a *generic* browser build: browser resolve conditions, no
/// aliases, and none of Vite's conventions. `vite: true` opts in to Vite
/// compatibility as a bundle — evaluating `vite.config` for `define`/`base`,
/// loading the `.env`/`VITE_*` file stack, injecting `import.meta.env`, and the
/// `NODE_ENV` production define. Vite behavior is never applied implicitly:
/// a project that wants it asks for it (`--vite`).
pub fn derive_web_config(root: &Path, vite: bool) -> Result<WebConfig, String> {
    let root = root
        .canonicalize()
        .map_err(|error| format!("cannot open project root {}: {error}", root.display()))?;
    let root = root.as_path();
    let conditions = ["module", "browser"]
        .iter()
        .map(|condition| condition.to_string())
        .collect();
    let mut config = WebConfig {
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
        vite: false,
        inputs: Vec::new(),
        emit_manifest: false,
        manifest_name: ".vite/manifest.json".to_string(),
        proxy: Vec::new(),
        optimize_deps_exclude: Vec::new(),
        out_dir: None,
    };
    if !vite {
        return Ok(config);
    }
    config.vite = true;
    // Vite mode: the same opt-in evaluation `build-app` uses — the config file's
    // computed `define` and `base` — with the same non-fatal fallback.
    let resolved = match diffpack_vite_compat::vite_config::resolve(root, "production") {
        Ok(resolved) => resolved,
        Err(error) => {
            eprintln!(
                "warning: could not evaluate vite config ({error}); continuing with defaults"
            );
            None
        }
    };
    let base = resolved
        .as_ref()
        .and_then(|resolved| resolved.base.clone())
        .unwrap_or_else(|| "/".to_string());
    let alias = resolved
        .as_ref()
        .map(|resolved| resolved.alias.clone())
        .unwrap_or_default();
    // `css.preprocessorOptions.scss.additionalData` (string form), threaded to
    // the native Sass compiler exactly like `base` and `define`.
    config.build.scss.additional_data = resolved
        .as_ref()
        .and_then(|resolved| resolved.scss_additional_data.clone());
    // `esbuild.*` / `oxc.jsx`, layered over each file's owning tsconfig.
    config.build.jsx = resolved
        .as_ref()
        .map(|resolved| resolved.jsx.clone())
        .unwrap_or_default();
    let mut defines = resolved
        .as_ref()
        .map(|resolved| resolved.define.clone())
        .unwrap_or_default();
    set_node_env(&mut defines, "production");
    // Vite resolves with the mode condition alongside the browser ones.
    config.build.conditions.push("production".to_string());
    // `resolve.conditions`: Vite ADDS user conditions to the environment defaults
    // (they widen, never replace), so append any not already present.
    if let Some(extra) = resolved
        .as_ref()
        .map(|resolved| &resolved.resolve_conditions)
    {
        for condition in extra {
            if !config
                .build
                .conditions
                .iter()
                .any(|existing| existing == condition)
            {
                config.build.conditions.push(condition.clone());
            }
        }
    }
    // `resolve.mainFields`: override the per-target default when the config sets it.
    if let Some(main_fields) = resolved
        .as_ref()
        .map(|resolved| resolved.main_fields.clone())
    {
        config.build.main_fields = main_fields;
    }
    // Vite normalizes `base` to end with `/`; URL joins depend on it.
    let base = if base.ends_with('/') {
        base
    } else {
        format!("{base}/")
    };
    config.build.base = base.clone();
    // Vite's default `assetsInlineLimit`.
    config.build.asset_inline_limit = 4096;
    // `resolve.alias` string finds, applied with Vite's exact-or-prefix
    // semantics by the resolver. A replacement that starts with `/` and does
    // not exist as a real absolute path is project-root-relative (Vite's
    // `'@': '/src'` convention).
    config.build.aliases = alias
        .into_iter()
        .map(|(find, replacement)| {
            let resolved = if let Some(rest) = replacement.strip_prefix('/')
                && !Path::new(&replacement).exists()
                && root.join(rest).exists()
            {
                root.join(rest).to_string_lossy().into_owned()
            } else {
                replacement
            };
            (find, resolved)
        })
        .collect();
    // `import.meta.env` from the full Vite source order: the `.env` file stack
    // for the mode, overridden by real `VITE_*` process variables (the overlay
    // is inside `load_vite_env`).
    let vite_vars = diffpack_vite_compat::env_file::load_vite_env(root, "production")?;
    let import_meta_env = Some(diffpack_vite_compat::import_meta_env::ImportMetaEnv {
        base: base.clone(),
        mode: "production".to_string(),
        vite_vars,
    });
    // `import.meta.glob` is part of the same Vite-convention opt-in; `/`-prefixed
    // patterns resolve against the project root, as in Vite.
    let import_meta_glob = Some(diffpack_vite_compat::import_meta_glob::ImportMetaGlob {
        root: root.to_path_buf(),
    });
    config.build.source_policy =
        std::sync::Arc::new(diffpack_vite_compat::source_policy::ViteSourcePolicy {
            import_meta_env,
            import_meta_glob,
            defines,
        });
    // `resolve.dedupe`: force each listed package to a single copy from the project
    // root's `node_modules`, exactly as Vite does — a directory alias the resolver
    // then resolves the package entry within. Appended AFTER the config aliases so it
    // does not clobber them; a package not present at the root is skipped. Only added
    // when the package is not already aliased, so an explicit alias wins.
    if let Some(dedupe) = resolved.as_ref().map(|resolved| resolved.dedupe.clone()) {
        for package in dedupe {
            if config
                .build
                .aliases
                .iter()
                .any(|(find, _)| *find == package)
            {
                continue;
            }
            let target = root.join("node_modules").join(&package);
            if target.is_dir() {
                config
                    .build
                    .aliases
                    .push((package, target.to_string_lossy().into_owned()));
            }
        }
    }
    // Multi-page inputs, manifest flag, dev proxy, and optimizeDeps.exclude for the
    // caller (`diffpack build` / `diffpack dev`).
    if let Some(resolved) = resolved.as_ref() {
        config.inputs = resolved
            .inputs
            .iter()
            .map(|(name, path)| (name.clone(), PathBuf::from(path)))
            .collect();
        config.emit_manifest = resolved.manifest;
        if let Some(name) = &resolved.manifest_name {
            config.manifest_name = name.clone();
        }
        config.proxy = resolved.proxy.clone();
        config.optimize_deps_exclude = resolved.optimize_deps_exclude.clone();
        // `build.outDir`, resolved against the project root exactly as Vite does.
        // `Path::join` with an absolute argument yields that path unchanged, so an
        // absolute `outDir` is honored as written.
        config.out_dir = resolved.out_dir.as_ref().map(|out_dir| root.join(out_dir));
        // `build.assetsDir` is NOT implemented: the emitters write hashed assets to
        // `assets/` unconditionally. Honoring the default silently is fine; a
        // non-default value would put every asset in the wrong place and every
        // emitted URL would 404, so say so instead of shipping that.
        if let Some(assets_dir) = resolved.assets_dir.as_deref()
            && assets_dir.trim_matches('/') != "assets"
        {
            return Err(format!(
                "vite config sets build.assetsDir = {assets_dir:?}, which diffpack does not \
                 implement: it emits hashed assets to `assets/` unconditionally, so every \
                 asset URL in the build would be wrong. Remove build.assetsDir (or set it to \
                 \"assets\") to build this project with diffpack."
            ));
        }
    }
    config.base = base;
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
fn set_node_env(defines: &mut Vec<(String, String)>, mode: &str) {
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

/// Reads the string value of a `key: '<value>'` (or `"<value>"`) option out of
/// `vite.config.ts`, if present. The single reader for every scalar option
/// Diffpack derives from the Vite config (`srcDirectory`, `routesDirectory`,
/// `routeFileIgnorePattern`, ...), so the parse lives in exactly one place. It is
/// a plain text read of a quoted literal, not a dependency on Vite: it does not
/// evaluate the config, so a value built from an expression is simply not found
/// (the caller falls back to the convention default).
pub fn vite_config_string(root: &Path, key: &str) -> Option<String> {
    diffpack_vite_compat::vite_config::config_string(root, key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    /// `vite.config`'s `build.outDir` must reach the caller, resolved against the
    /// project root exactly as Vite resolves it. The defect was that `WebConfig`
    /// had no such field at all, so an app configuring `build: { outDir: "build" }`
    /// silently got `dist/` — a build that "succeeded" into the wrong directory.
    #[test]
    fn vite_build_out_dir_is_read_and_resolved_against_the_project_root() {
        if Command::new("node").arg("--version").output().is_err() {
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        std::fs::write(root.join("index.html"), "<!doctype html><html></html>").unwrap();
        std::fs::write(
            root.join("vite.config.mjs"),
            "export default { build: { outDir: 'build' } };",
        )
        .unwrap();
        let config = derive_web_config(root, true).unwrap();
        assert_eq!(
            config.out_dir,
            Some(root.canonicalize().unwrap().join("build"))
        );

        // With no `build.outDir`, the field stays `None` and the caller applies
        // Vite's `dist` default — an unset value must not be forged into one here.
        std::fs::write(root.join("vite.config.mjs"), "export default {};").unwrap();
        assert_eq!(derive_web_config(root, true).unwrap().out_dir, None);
    }

    /// `build.assetsDir` is NOT implemented — the emitters hardcode `assets/`.
    /// A non-default value must therefore stop the build by name, not produce an
    /// output whose every asset URL is wrong. The default value is accepted,
    /// because honoring it is exactly what diffpack already does.
    #[test]
    fn a_non_default_vite_assets_dir_is_a_named_error_not_a_silent_wrong_output() {
        if Command::new("node").arg("--version").output().is_err() {
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        std::fs::write(root.join("index.html"), "<!doctype html><html></html>").unwrap();
        std::fs::write(
            root.join("vite.config.mjs"),
            "export default { build: { assetsDir: 'static' } };",
        )
        .unwrap();
        let error = derive_web_config(root, true).unwrap_err();
        assert!(error.contains("build.assetsDir"), "{error}");
        assert!(error.contains("\"static\""), "{error}");

        std::fs::write(
            root.join("vite.config.mjs"),
            "export default { build: { assetsDir: 'assets' } };",
        )
        .unwrap();
        assert!(derive_web_config(root, true).is_ok());
    }
}
