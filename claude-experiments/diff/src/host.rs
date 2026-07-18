//! Plugin-host bridge.
//!
//! Diffpack owns the module graph, linking, chunking, and emit. A Node sidecar
//! (`host/sidecar.mjs`) answers the build-time questions that require the
//! project's own JavaScript plugins. Today that is configuration: the resolver
//! aliases the framework plugin computes (e.g. TanStack's
//! `#tanstack-router-entry` -> the app router). Config is fetched **once per
//! build**, so this never touches the per-edit incremental path and cannot
//! regress the incrementality guards.
//!
//! The sidecar script is embedded (`include_str!`) and materialized to a temp
//! file at run time, so the binary carries its own host without a path
//! dependency, and the guest code still lives in a real `.mjs` file.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::bundler::BuildConfig;

const SIDECAR_SOURCE: &str = include_str!("../host/sidecar.mjs");

/// Resolved plugin-host configuration for one build environment.
#[derive(Debug, Clone)]
pub struct HostConfig {
    /// The environment this config is for (`client`, `ssr`, ...).
    pub environment: String,
    /// All environments the project's config defines.
    pub environments: Vec<String>,
    /// Bundler configuration derived from the host (currently the resolver
    /// aliases).
    pub build: BuildConfig,
    /// Non-string (regex/function) aliases the sidecar could not report. Surfaced
    /// so a silent drop is visible rather than mistaken for "none".
    pub skipped_aliases: usize,
}

/// Runs the sidecar to resolve the project's build config for `environment`.
pub fn resolve_config(project_root: &Path, environment: &str) -> Result<HostConfig, String> {
    // The sidecar anchors module resolution at the root via `createRequire`,
    // which needs an absolute path.
    let project_root = project_root
        .canonicalize()
        .map_err(|error| format!("cannot open project root {}: {error}", project_root.display()))?;
    let sidecar = materialize_sidecar()?;
    let output = Command::new("node")
        .arg(&sidecar)
        .arg("resolve-config")
        .arg(&project_root)
        .arg(environment)
        .output()
        .map_err(|error| format!("cannot run the plugin-host sidecar (is node installed?): {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "sidecar resolve-config failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    let response: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("sidecar returned invalid JSON: {error}"))?;

    let environments = string_array(&response["environments"], "environments")?;
    let resolved_environment = response["environment"]
        .as_str()
        .ok_or("sidecar response missing environment")?
        .to_string();
    let skipped_aliases = response["skippedAliases"].as_u64().unwrap_or(0) as usize;

    let aliases = response["aliases"]
        .as_array()
        .ok_or("sidecar response missing aliases array")?
        .iter()
        .map(|pair| {
            let find = pair
                .get(0)
                .and_then(serde_json::Value::as_str)
                .ok_or("alias `find` is not a string")?;
            let replacement = pair
                .get(1)
                .and_then(serde_json::Value::as_str)
                .ok_or("alias `replacement` is not a string")?;
            Ok((find.to_string(), replacement.to_string()))
        })
        .collect::<Result<Vec<_>, String>>()?;

    Ok(HostConfig {
        environment: resolved_environment,
        environments,
        build: BuildConfig { aliases },
        skipped_aliases,
    })
}

fn string_array(value: &serde_json::Value, field: &str) -> Result<Vec<String>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("sidecar response field {field:?} is not an array"))?
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("sidecar response field {field:?} has a non-string entry"))
        })
        .collect()
}

/// Writes the embedded sidecar to a stable temp path (named by content hash, so
/// concurrent builds share it and it is rewritten only when the source changes).
fn materialize_sidecar() -> Result<PathBuf, String> {
    let hash = fnv1a(SIDECAR_SOURCE.as_bytes());
    let path = std::env::temp_dir().join(format!("diffpack-sidecar-{hash:016x}.mjs"));
    let needs_write = match fs::read(&path) {
        Ok(existing) => existing != SIDECAR_SOURCE.as_bytes(),
        Err(_) => true,
    };
    if needs_write {
        fs::write(&path, SIDECAR_SOURCE)
            .map_err(|error| format!("cannot write sidecar to {}: {error}", path.display()))?;
    }
    Ok(path)
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
