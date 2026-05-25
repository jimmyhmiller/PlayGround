//! Host-wiring portion of a program manifest. The runtime/IR side is
//! parsed by `ir::load_manifest_file`; this struct picks up the
//! `[adapters]` section from the same file (which the IR parser ignores as
//! unknown). Generators live in the manifest proper (`[generators.<name>]`)
//! since they're a typed program-level concept.

use std::path::PathBuf;

use indexmap::IndexMap;
use serde::Deserialize;
use toml::Value as TomlValue;

/// Runner-only sections of the program file.
#[derive(Debug, Deserialize)]
pub struct RunnerConfig {
    /// Directory the runner searches for handler-body WASM components.
    /// Defaults to the directory containing the program file. Paths are
    /// tried in two conventions: `<dir>/<uri>.component.wasm` (flat) and
    /// `<dir>/<uri>/<uri>.component.wasm` (nested).
    #[serde(default)]
    pub components_dir: Option<PathBuf>,

    /// One entry per declared effect in the manifest.
    #[serde(default)]
    pub adapters: IndexMap<String, AdapterConfig>,
}

#[derive(Debug, Deserialize)]
pub struct AdapterConfig {
    pub kind: String,
    #[serde(flatten, default)]
    pub options: IndexMap<String, TomlValue>,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("toml: {0}")]
    Toml(#[from] toml::de::Error),
}

pub fn load(path: &std::path::Path) -> Result<RunnerConfig, ConfigError> {
    let raw = std::fs::read_to_string(path)?;
    Ok(toml::from_str(&raw)?)
}
