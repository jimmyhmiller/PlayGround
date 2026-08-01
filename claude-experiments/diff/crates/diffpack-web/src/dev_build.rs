//! Long-lived incremental build state shared by development-server adapters.

use std::collections::BTreeSet;
use std::path::Path;
use std::path::PathBuf;

use diffpack_default_loader::driver::{
    Bundler, DirectReachability, EmitOptions, partition_diagnostics,
};

/// Framework-neutral options shared by development-server adapters.
pub struct DevOptions {
    pub project_root: std::path::PathBuf,
    pub port: u16,
    pub minify: bool,
    pub source_map: bool,
}

/// One long-lived environment build (client or server).
///
/// Keeping the bundler and reachability session alive across edits makes rebuilds
/// incremental while still exposing the current reachable set to each adapter's
/// output and manifest logic.
pub struct EnvBuild {
    pub bundler: Bundler,
    pub session: DirectReachability,
    pub reachable: BTreeSet<String>,
    pub options: EmitOptions,
}

impl EnvBuild {
    pub fn reachable_ids(&self) -> BTreeSet<String> {
        self.reachable.clone()
    }

    /// Rebuild after `path` changed and apply its reachability delta.
    pub fn rebuild(&mut self, path: &Path) -> Result<Rebuilt, String> {
        let update = self.bundler.rebuild_path(path)?;
        let transformed = update.transformed_modules;
        let changed = update.delta.changed.len();
        let result = self.session.apply(&update.delta);
        let graph_changed = !result.added.is_empty() || !result.removed.is_empty();
        for module in result.removed {
            self.reachable.remove(&module);
        }
        self.reachable.extend(result.added);
        for warning in partition_diagnostics(&update.diagnostics, "rebuild")? {
            eprintln!("[dev] warning: {warning}");
        }
        Ok(Rebuilt {
            transformed,
            changed,
            changed_ids: update.delta.changed.clone(),
            graph_changed,
        })
    }
}

/// Derive a compact set of watch roots from the modules actually compiled by
/// several live environments.
pub fn source_watch_roots(
    project_root: &Path,
    envs: &[&EnvBuild],
) -> Vec<(PathBuf, notify::RecursiveMode)> {
    let mut directories: BTreeSet<PathBuf> = BTreeSet::new();
    for env in envs {
        for id in &env.reachable {
            let path = Path::new(id.split('?').next().unwrap_or(id.as_str()));
            if !path.is_absolute() {
                continue;
            }
            let excluded = path.components().any(|component| {
                let name = component.as_os_str();
                name == "node_modules" || name.as_encoded_bytes().starts_with(b".")
            });
            if excluded {
                continue;
            }
            if let Some(parent) = path.parent() {
                directories.insert(parent.to_path_buf());
            }
        }
    }
    if directories.is_empty() {
        return vec![(project_root.to_path_buf(), notify::RecursiveMode::Recursive)];
    }

    let mut common = directories
        .iter()
        .cloned()
        .reduce(|a, b| {
            a.ancestors()
                .find(|ancestor| b.starts_with(ancestor))
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("/"))
        })
        .unwrap_or_else(|| project_root.to_path_buf());
    let floor = project_root.parent().unwrap_or(project_root);
    if !floor.starts_with(&common) || common.components().count() < 2 {
        common = project_root.to_path_buf();
    }

    let mut roots: BTreeSet<PathBuf> = BTreeSet::new();
    for directory in &directories {
        match directory.strip_prefix(&common) {
            Ok(relative) => {
                if let Some(first) = relative.components().next() {
                    roots.insert(common.join(first));
                }
            }
            Err(_) => {
                roots.insert(directory.clone());
            }
        }
    }
    let mut out = vec![(common, notify::RecursiveMode::NonRecursive)];
    out.extend(
        roots
            .into_iter()
            .map(|root| (root, notify::RecursiveMode::Recursive)),
    );
    out
}

/// Aggregated per-edit counters for one environment across a coalesced batch.
#[derive(Default)]
pub struct EnvCounters {
    pub transformed: usize,
    pub changed: usize,
    pub rendered_chunks: usize,
}

impl EnvCounters {
    pub fn add(&mut self, rebuilt: &Rebuilt, rendered_chunks: usize) {
        self.transformed += rebuilt.transformed;
        self.changed += rebuilt.changed;
        self.rendered_chunks += rendered_chunks;
    }
}

/// Per-edit rebuild counts for one environment.
#[derive(Default)]
pub struct Rebuilt {
    pub transformed: usize,
    pub changed: usize,
    pub changed_ids: BTreeSet<String>,
    pub graph_changed: bool,
}
