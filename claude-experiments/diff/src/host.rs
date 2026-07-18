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
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

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

    let conditions = string_array(&response["conditions"], "conditions").unwrap_or_default();

    Ok(HostConfig {
        environment: resolved_environment,
        environments,
        build: BuildConfig {
            aliases,
            conditions,
        },
        skipped_aliases,
    })
}

/// A long-lived sidecar process that answers `resolveId`/`load` for framework
/// virtual and plugin-generated modules. Diffpack calls it only for ids its own
/// resolver/loader cannot handle, so the common path stays native. Requests are
/// serialized (one in flight) behind a mutex; each is cheap relative to the JS
/// hook it runs, and it is never on the per-module hot path for ordinary files.
pub struct Sidecar {
    inner: Mutex<SidecarProcess>,
}

struct SidecarProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl Sidecar {
    /// Spawns `node sidecar.mjs serve <project_root>`.
    pub fn start(project_root: &Path) -> Result<Self, String> {
        let project_root = project_root.canonicalize().map_err(|error| {
            format!("cannot open project root {}: {error}", project_root.display())
        })?;
        let sidecar = materialize_sidecar()?;
        let mut child = Command::new("node")
            .arg(&sidecar)
            .arg("serve")
            .arg(&project_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| format!("cannot start the plugin-host sidecar (is node installed?): {error}"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or("sidecar stdin was not captured")?;
        let stdout = child
            .stdout
            .take()
            .ok_or("sidecar stdout was not captured")?;
        Ok(Self {
            inner: Mutex::new(SidecarProcess {
                child,
                stdin,
                stdout: BufReader::new(stdout),
                next_id: 1,
            }),
        })
    }

    fn request(&self, mut payload: serde_json::Value) -> Result<serde_json::Value, String> {
        let mut process = self.inner.lock().map_err(|_| "sidecar mutex poisoned")?;
        let id = process.next_id;
        process.next_id += 1;
        payload["id"] = serde_json::Value::from(id);
        let mut line = serde_json::to_string(&payload).map_err(|error| error.to_string())?;
        line.push('\n');
        process
            .stdin
            .write_all(line.as_bytes())
            .and_then(|()| process.stdin.flush())
            .map_err(|error| format!("cannot write to sidecar: {error}"))?;

        let mut response_line = String::new();
        let read = process
            .stdout
            .read_line(&mut response_line)
            .map_err(|error| format!("cannot read from sidecar: {error}"))?;
        if read == 0 {
            return Err("sidecar closed its output stream".to_string());
        }
        let response: serde_json::Value = serde_json::from_str(response_line.trim())
            .map_err(|error| format!("sidecar returned invalid JSON: {error}"))?;
        if let Some(error) = response.get("error").and_then(serde_json::Value::as_str) {
            return Err(format!("sidecar error: {error}"));
        }
        Ok(response)
    }

    /// Asks a framework plugin to resolve `specifier`. Returns the resolved
    /// (possibly virtual) module id, or `None` if no plugin claims it.
    pub fn resolve_id(
        &self,
        environment: &str,
        specifier: &str,
        importer: Option<&str>,
    ) -> Result<Option<String>, String> {
        let response = self.request(serde_json::json!({
            "op": "resolveId",
            "environment": environment,
            "specifier": specifier,
            "importer": importer,
        }))?;
        Ok(response
            .get("resolved")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string))
    }

    /// Asks a framework plugin to load a (virtual) module id. Returns its source,
    /// or `None` if no plugin provides it.
    pub fn load(&self, environment: &str, module_id: &str) -> Result<Option<String>, String> {
        let response = self.request(serde_json::json!({
            "op": "load",
            "environment": environment,
            "moduleId": module_id,
        }))?;
        Ok(response
            .get("code")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string))
    }

    /// Asks the sidecar to shut down cleanly.
    pub fn shutdown(&self) {
        let _ = self.request(serde_json::json!({ "op": "shutdown" }));
    }
}

impl Drop for Sidecar {
    fn drop(&mut self) {
        if let Ok(mut process) = self.inner.lock() {
            let _ = process.child.kill();
            let _ = process.child.wait();
        }
    }
}

/// A running sidecar bound to one build environment, handed to the bundler so it
/// can fall back to framework `resolveId`/`load` for ids it cannot resolve or
/// load itself (virtual and plugin-generated modules). The bundler holds it for
/// the lifetime of the build, so incremental rebuilds reuse the same warm host.
pub struct HostBridge {
    sidecar: Sidecar,
    environment: String,
}

impl HostBridge {
    pub fn new(sidecar: Sidecar, environment: String) -> Self {
        Self { sidecar, environment }
    }

    /// Resolves a specifier no native rule handled. Returns a (virtual) module id
    /// or `None`.
    pub fn resolve_id(&self, specifier: &str, importer: Option<&str>) -> Option<String> {
        self.sidecar
            .resolve_id(&self.environment, specifier, importer)
            .ok()
            .flatten()
    }

    /// Loads a (virtual) module id no native loader handled. Returns its source or
    /// `None`.
    pub fn load(&self, module_id: &str) -> Option<String> {
        self.sidecar.load(&self.environment, module_id).ok().flatten()
    }
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
