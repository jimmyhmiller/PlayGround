//! Filesystem output primitives shared by emitters and integrations.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Integration-owned files layered onto a generic emitted environment.
pub trait OutputIntegrationPolicy: Send + Sync {
    fn write_server_runtime(&self, _server_dir: &Path, _hmr: bool) -> Result<Vec<PathBuf>, String> {
        Ok(Vec::new())
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NoOutputIntegrationPolicy;

impl OutputIntegrationPolicy for NoOutputIntegrationPolicy {}

/// Ordered composition of integration-owned output writers.
#[derive(Default)]
pub struct OutputIntegrationPolicyChain {
    policies: Vec<Arc<dyn OutputIntegrationPolicy>>,
}

impl OutputIntegrationPolicyChain {
    pub fn new(policies: Vec<Arc<dyn OutputIntegrationPolicy>>) -> Self {
        Self { policies }
    }
}

impl OutputIntegrationPolicy for OutputIntegrationPolicyChain {
    fn write_server_runtime(&self, server_dir: &Path, hmr: bool) -> Result<Vec<PathBuf>, String> {
        let mut written = Vec::new();
        for policy in &self.policies {
            written.extend(policy.write_server_runtime(server_dir, hmr)?);
        }
        Ok(written)
    }
}

/// Deletes stale files below an output root and removes directories made empty.
pub fn prune_output(root: &Path, keep: &BTreeSet<PathBuf>) -> Result<(), String> {
    if !root.exists() {
        return Ok(());
    }
    let mut directories = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(directory) = stack.pop() {
        directories.push(directory.clone());
        let entries = fs::read_dir(&directory)
            .map_err(|error| format!("cannot read {}: {error}", directory.display()))?;
        for entry in entries {
            let entry =
                entry.map_err(|error| format!("cannot read {}: {error}", directory.display()))?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if !keep.contains(&path) {
                fs::remove_file(&path)
                    .map_err(|error| format!("cannot remove {}: {error}", path.display()))?;
            }
        }
    }
    directories.sort_by_key(|directory| std::cmp::Reverse(directory.components().count()));
    for directory in directories {
        if directory != root
            && fs::read_dir(&directory)
                .map(|mut entries| entries.next().is_none())
                .unwrap_or(false)
        {
            fs::remove_dir(&directory)
                .map_err(|error| format!("cannot remove {}: {error}", directory.display()))?;
        }
    }
    Ok(())
}

/// Counts the files that actually landed in one emitted environment directory.
#[derive(Debug, Clone)]
pub struct EmitSummary {
    pub output_dir: PathBuf,
    pub javascript_files: usize,
    pub css_files: usize,
    pub asset_files: usize,
    pub rendered_chunks: usize,
}

impl EmitSummary {
    pub fn of(output_dir: &Path) -> Result<Self, String> {
        let mut summary = Self {
            output_dir: output_dir.to_path_buf(),
            javascript_files: 0,
            css_files: 0,
            asset_files: 0,
            rendered_chunks: 0,
        };
        let mut stack = vec![output_dir.to_path_buf()];
        while let Some(directory) = stack.pop() {
            let entries = fs::read_dir(&directory)
                .map_err(|error| format!("cannot read {}: {error}", directory.display()))?;
            for entry in entries {
                let entry = entry
                    .map_err(|error| format!("cannot read {}: {error}", directory.display()))?;
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path
                    .parent()
                    .and_then(Path::file_name)
                    .and_then(|name| name.to_str())
                    == Some("assets")
                {
                    summary.asset_files += 1;
                } else {
                    match path.extension().and_then(|value| value.to_str()) {
                        Some("js" | "mjs") => summary.javascript_files += 1,
                        Some("css") => summary.css_files += 1,
                        _ => {}
                    }
                }
            }
        }
        Ok(summary)
    }
}

/// Writes `bytes` only when the destination does not already contain them.
/// This preserves mtimes for cache-reused chunks and generated files.
pub fn write_if_changed(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Ok(existing) = fs::read(path)
        && existing == bytes
    {
        return Ok(());
    }
    fs::write(path, bytes).map_err(|error| format!("cannot write {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    #[test]
    fn identical_output_preserves_the_destination() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("chunk.js");
        write_if_changed(&path, b"first").unwrap();
        let modified = fs::metadata(&path)
            .unwrap()
            .modified()
            .unwrap_or(SystemTime::UNIX_EPOCH);

        write_if_changed(&path, b"first").unwrap();

        assert_eq!(fs::read(&path).unwrap(), b"first");
        assert_eq!(fs::metadata(&path).unwrap().modified().unwrap(), modified);
    }

    #[test]
    fn summary_counts_environment_outputs() {
        let directory = tempfile::tempdir().unwrap();
        fs::create_dir(directory.path().join("assets")).unwrap();
        fs::write(directory.path().join("index.js"), "").unwrap();
        fs::write(directory.path().join("index.css"), "").unwrap();
        fs::write(directory.path().join("assets/logo.svg"), "").unwrap();

        let summary = EmitSummary::of(directory.path()).unwrap();
        assert_eq!(summary.javascript_files, 1);
        assert_eq!(summary.css_files, 1);
        assert_eq!(summary.asset_files, 1);
    }

    #[test]
    fn pruning_preserves_live_files_and_removes_empty_directories() {
        let root = tempfile::tempdir().unwrap();
        let live = root.path().join("assets/live.js");
        let stale = root.path().join("old/stale.js");
        fs::create_dir_all(live.parent().unwrap()).unwrap();
        fs::create_dir_all(stale.parent().unwrap()).unwrap();
        fs::write(&live, "live").unwrap();
        fs::write(&stale, "stale").unwrap();

        prune_output(root.path(), &BTreeSet::from([live.clone()])).unwrap();

        assert!(live.is_file());
        assert!(!stale.exists());
        assert!(!root.path().join("old").exists());
        assert!(root.path().is_dir());
    }
}
