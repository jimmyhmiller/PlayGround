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

/// The subset of a resolved Vite config Diffpack consumes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResolvedViteConfig {
    /// `base` (`import.meta.env.BASE_URL`), if the config sets it as a string.
    pub base: Option<String>,
    /// `define` entries as `(identifier, replacement_source)`, already normalized
    /// to raw replacement text (a string value verbatim, else JSON-stringified).
    pub define: Vec<(String, String)>,
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

    Ok(ResolvedViteConfig { base, define })
}
