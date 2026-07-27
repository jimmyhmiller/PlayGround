//! One-time evaluation of a Vite config file.
//!
//! A `vite.config.ts` is a program, not data: its values can be computed
//! (`JSON.stringify(pkg.version)`, `loadEnv(...)`, expressions), so they cannot be
//! read as text. Vite itself evaluates the config in Node before handing resolved
//! values to its bundler; Diffpack does the same, spawning `node` **once** to
//! evaluate the config and return the fields it needs as JSON. This is the only
//! place Diffpack invokes Node, and it is not the build path: the entire build
//! (graph, transform, chunk, emit) is native Rust. It mirrors exactly how Vite
//! (and rolldown-vite, whose Rust engine is still driven by a Node process)
//! separates config evaluation from bundling.
//!
//! Failure is non-fatal: if `node` is absent or the config cannot be evaluated,
//! the caller proceeds with convention defaults (and a surfaced warning) rather
//! than aborting the build.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The evaluator script, kept as a real `.mjs` file and embedded, run via `node`'s
/// stdin so no temporary file is written.
const EVALUATOR: &str = include_str!("vite_config_evaluator.mjs");

/// One `server.proxy` rule: forward requests whose path begins with `context` to
/// `target`. `change_origin` rewrites the forwarded `Host` header to the target's
/// host (Vite's `changeOrigin`); `ws` marks the rule as also proxying WebSocket
/// upgrades. A `rewrite` FUNCTION cannot be expressed natively; the evaluator
/// counts such rules and [`resolve`] surfaces a warning, never a silent drop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProxyRule {
    pub context: String,
    pub target: String,
    pub change_origin: bool,
    pub ws: bool,
}

/// The subset of a resolved Vite config Diffpack consumes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResolvedViteConfig {
    /// `base` (`import.meta.env.BASE_URL`), if the config sets it as a string.
    pub base: Option<String>,
    /// `define` entries as `(identifier, replacement_source)`, already normalized
    /// to raw replacement text (a string value verbatim, else JSON-stringified).
    pub define: Vec<(String, String)>,
    /// `resolve.alias` entries with string finds, as `(find, replacement)`.
    /// Regex/function finds cannot be expressed to the native resolver; the
    /// evaluator counts them and [`resolve`] surfaces a warning, never a
    /// silent drop.
    pub alias: Vec<(String, String)>,
    /// `css.preprocessorOptions.scss.additionalData` when it is a string:
    /// prepended to every compiled `.scss` root, exactly as Vite does. A
    /// function value cannot be expressed natively; the evaluator counts it
    /// and [`resolve`] surfaces a warning, never a silent drop.
    pub scss_additional_data: Option<String>,
    /// `build.rollupOptions.input` normalized to ordered `(name, absolute_path)`
    /// pairs — the multi-page entry set. Empty means the single-`index.html`
    /// default. Each path is absolute (resolved against the project root by the
    /// evaluator).
    pub inputs: Vec<(String, String)>,
    /// `build.manifest`: whether to emit the build manifest, and its file name
    /// (Vite's default is `.vite/manifest.json`).
    pub manifest: bool,
    pub manifest_name: Option<String>,
    /// `resolve.conditions`: extra export-map conditions to honor, added to the
    /// environment defaults.
    pub resolve_conditions: Vec<String>,
    /// `resolve.mainFields`: the `package.json` fields to try, in order. Empty
    /// keeps the built-in per-target default.
    pub main_fields: Vec<String>,
    /// `resolve.dedupe`: packages that must resolve to a single copy from the
    /// project root's `node_modules` (applied as root-directory aliases).
    pub dedupe: Vec<String>,
    /// `optimizeDeps.exclude`: dependencies excluded from pre-bundling. Diffpack
    /// bundles every dependency natively, so this is satisfied by construction;
    /// surfaced for honest reporting.
    pub optimize_deps_exclude: Vec<String>,
    /// `server.proxy` rules for the dev server.
    pub proxy: Vec<ProxyRule>,
}

/// The candidate config filenames, in Vite's resolution order.
const CONFIG_FILES: [&str; 4] = [
    "vite.config.ts",
    "vite.config.mts",
    "vite.config.js",
    "vite.config.mjs",
];

/// Locates the project's Vite config file, if any.
pub fn config_file(root: &Path) -> Option<PathBuf> {
    CONFIG_FILES
        .iter()
        .map(|name| root.join(name))
        .find(|path| path.is_file())
}

/// Evaluates the project's Vite config in `mode` and returns the fields Diffpack
/// needs. `Ok(None)` when there is no config file. An `Err` carries a message the
/// caller should surface (then fall back to defaults); it never means the build
/// must stop.
pub fn resolve(root: &Path, mode: &str) -> Result<Option<ResolvedViteConfig>, String> {
    let Some(config_path) = config_file(root) else {
        return Ok(None);
    };

    let mut child = Command::new("node")
        .arg("--input-type=module")
        .arg("--no-warnings")
        .env("DIFFPACK_VITE_CONFIG", &config_path)
        .env("DIFFPACK_VITE_MODE", mode)
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("cannot spawn `node` to evaluate {}: {error}", config_path.display()))?;

    // Write the evaluator to stdin and close it, so `node` sees EOF and runs before
    // we block on its output (avoids a pipe deadlock).
    {
        let mut stdin = child
            .stdin
            .take()
            .ok_or("failed to open node stdin for the vite-config evaluator")?;
        stdin
            .write_all(EVALUATOR.as_bytes())
            .map_err(|error| format!("cannot pipe the vite-config evaluator to node: {error}"))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|error| format!("node (vite-config evaluator) did not complete: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "evaluating {} in node failed: {}",
            config_path.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    parse(&output.stdout).map(Some)
}

/// Parses the evaluator's JSON output into a [`ResolvedViteConfig`]. A hand parse
/// keeps this dependency-light and shapes the `define` map into ordered pairs.
fn parse(stdout: &[u8]) -> Result<ResolvedViteConfig, String> {
    let value: serde_json::Value = serde_json::from_slice(stdout)
        .map_err(|error| format!("vite-config evaluator produced invalid JSON: {error}"))?;

    let base = value
        .get("base")
        .and_then(|base| base.as_str())
        .map(str::to_string);

    let mut define = Vec::new();
    if let Some(serde_json::Value::Object(map)) = value.get("define") {
        for (key, replacement) in map {
            if let Some(replacement) = replacement.as_str() {
                define.push((key.clone(), replacement.to_string()));
            }
        }
    }
    // Deterministic order (the graph transform must be reproducible).
    define.sort();
    let mut alias = Vec::new();
    if let Some(serde_json::Value::Array(entries)) = value.get("alias") {
        for entry in entries {
            if let Some([find, replacement]) = entry.as_array().map(|pair| pair.as_slice())
                && let (Some(find), Some(replacement)) = (find.as_str(), replacement.as_str())
            {
                alias.push((find.to_string(), replacement.to_string()));
            }
        }
    }
    if let Some(skipped) = value.get("aliasSkipped").and_then(|value| value.as_u64())
        && skipped > 0
    {
        eprintln!(
            "warning: vite config has {skipped} resolve.alias entr{} with a regex/function \
             find, which diffpack cannot apply",
            if skipped == 1 { "y" } else { "ies" }
        );
    }
    let scss_additional_data = value
        .get("scssAdditionalData")
        .and_then(|data| data.as_str())
        .map(str::to_string);
    if value
        .get("scssAdditionalDataSkipped")
        .and_then(|value| value.as_u64())
        .is_some_and(|skipped| skipped > 0)
    {
        eprintln!(
            "warning: vite config sets css.preprocessorOptions.scss.additionalData to a \
             non-string value, which diffpack cannot apply"
        );
    }

    // `build.rollupOptions.input` -> ordered `(name, absolute_path)` pairs.
    let mut inputs = Vec::new();
    if let Some(serde_json::Value::Array(entries)) = value.get("inputs") {
        for entry in entries {
            if let Some([name, path]) = entry.as_array().map(|pair| pair.as_slice())
                && let (Some(name), Some(path)) = (name.as_str(), path.as_str())
            {
                inputs.push((name.to_string(), path.to_string()));
            }
        }
    }
    let manifest = value
        .get("manifest")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let manifest_name = value
        .get("manifestName")
        .and_then(|value| value.as_str())
        .map(str::to_string);
    let string_array = |key: &str| -> Vec<String> {
        match value.get(key) {
            Some(serde_json::Value::Array(entries)) => entries
                .iter()
                .filter_map(|entry| entry.as_str().map(str::to_string))
                .collect(),
            _ => Vec::new(),
        }
    };
    let resolve_conditions = string_array("resolveConditions");
    let main_fields = string_array("mainFields");
    let dedupe = string_array("dedupe");
    let optimize_deps_exclude = string_array("optimizeDepsExclude");

    let mut proxy = Vec::new();
    if let Some(serde_json::Value::Array(entries)) = value.get("proxy") {
        for entry in entries {
            let (Some(context), Some(target)) = (
                entry.get("context").and_then(|value| value.as_str()),
                entry.get("target").and_then(|value| value.as_str()),
            ) else {
                continue;
            };
            proxy.push(ProxyRule {
                context: context.to_string(),
                target: target.to_string(),
                change_origin: entry
                    .get("changeOrigin")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false),
                ws: entry
                    .get("ws")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false),
            });
        }
    }
    if value
        .get("proxyRewriteSkipped")
        .and_then(|value| value.as_u64())
        .is_some_and(|skipped| skipped > 0)
    {
        eprintln!(
            "warning: vite config has server.proxy rule(s) with a `rewrite` function, \
             which diffpack's native dev proxy cannot apply (the path is forwarded as-is)"
        );
    }

    Ok(ResolvedViteConfig {
        base,
        define,
        alias,
        scss_additional_data,
        inputs,
        manifest,
        manifest_name,
        resolve_conditions,
        main_fields,
        dedupe,
        optimize_deps_exclude,
        proxy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_multi_page_inputs_and_manifest() {
        let json = br#"{
            "inputs": [["main", "/abs/index.html"], ["about", "/abs/about.html"]],
            "manifest": true,
            "manifestName": ".vite/manifest.json"
        }"#;
        let resolved = parse(json).unwrap();
        assert_eq!(
            resolved.inputs,
            vec![
                ("main".to_string(), "/abs/index.html".to_string()),
                ("about".to_string(), "/abs/about.html".to_string()),
            ]
        );
        assert!(resolved.manifest);
        assert_eq!(resolved.manifest_name.as_deref(), Some(".vite/manifest.json"));
    }

    #[test]
    fn parses_resolve_and_optimize_deps_fields() {
        let json = br#"{
            "resolveConditions": ["custom", "browser"],
            "mainFields": ["module", "main"],
            "dedupe": ["react", "react-dom"],
            "optimizeDepsExclude": ["some-esm-only-dep"]
        }"#;
        let resolved = parse(json).unwrap();
        assert_eq!(resolved.resolve_conditions, vec!["custom", "browser"]);
        assert_eq!(resolved.main_fields, vec!["module", "main"]);
        assert_eq!(resolved.dedupe, vec!["react", "react-dom"]);
        assert_eq!(resolved.optimize_deps_exclude, vec!["some-esm-only-dep"]);
    }

    #[test]
    fn parses_server_proxy_string_and_object_forms() {
        let json = br#"{
            "proxy": [
                {"context": "/api", "target": "http://localhost:3001", "changeOrigin": false, "ws": false},
                {"context": "/socket", "target": "ws://localhost:4000", "changeOrigin": true, "ws": true}
            ]
        }"#;
        let resolved = parse(json).unwrap();
        assert_eq!(resolved.proxy.len(), 2);
        assert_eq!(resolved.proxy[0].context, "/api");
        assert_eq!(resolved.proxy[0].target, "http://localhost:3001");
        assert!(!resolved.proxy[0].change_origin);
        assert!(resolved.proxy[1].change_origin);
        assert!(resolved.proxy[1].ws);
    }

    #[test]
    fn absent_fields_default_empty() {
        let resolved = parse(b"{}").unwrap();
        assert!(resolved.inputs.is_empty());
        assert!(!resolved.manifest);
        assert!(resolved.resolve_conditions.is_empty());
        assert!(resolved.main_fields.is_empty());
        assert!(resolved.dedupe.is_empty());
        assert!(resolved.optimize_deps_exclude.is_empty());
        assert!(resolved.proxy.is_empty());
    }
}
