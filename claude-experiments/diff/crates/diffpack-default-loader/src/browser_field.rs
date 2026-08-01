//! Package-level policy for the object form of `package.json`'s `browser` field.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Memoizes whether directories may be remapped by their package's `browser`
/// field. The resolver uses this to decide whether its relative-path fast path
/// is safe.
pub struct BrowserFieldMap {
    honored: bool,
    directories: Mutex<HashMap<PathBuf, bool>>,
}

impl BrowserFieldMap {
    pub fn new(honored: bool) -> Self {
        Self {
            honored,
            directories: Mutex::new(HashMap::new()),
        }
    }

    pub fn remaps_directory(&self, directory: &Path) -> bool {
        if !self.honored {
            return false;
        }
        if let Some(known) = self
            .directories
            .lock()
            .expect("browser-field directory cache poisoned")
            .get(directory)
        {
            return *known;
        }
        let answer = nearest_package_has_object_browser_field(directory);
        self.directories
            .lock()
            .expect("browser-field directory cache poisoned")
            .insert(directory.to_path_buf(), answer);
        answer
    }
}

/// Whether the nearest package manifest governing `directory` carries an
/// object-form `browser` field.
pub fn nearest_package_has_object_browser_field(directory: &Path) -> bool {
    for ancestor in directory.ancestors() {
        let manifest = ancestor.join("package.json");
        if !manifest.is_file() {
            continue;
        }
        let Ok(text) = fs::read_to_string(&manifest) else {
            return true;
        };
        if !text.contains("\"browser\"") {
            return false;
        }
        let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&text) else {
            return true;
        };
        return manifest
            .get("browser")
            .is_some_and(serde_json::Value::is_object);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_policy_never_reads_package_policy() {
        assert!(!BrowserFieldMap::new(false).remaps_directory(Path::new(".")));
    }
}
