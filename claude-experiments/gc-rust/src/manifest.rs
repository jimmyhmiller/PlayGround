//! `gcr.toml` project manifest — a tiny, dependency-free reader for the handful
//! of keys gc-rust needs. A project is a directory containing `gcr.toml`; the
//! manifest names the package and points at its entry source file.
//!
//! ```toml
//! [package]
//! name = "myapp"
//! version = "0.1.0"
//! entry = "src/main.gcr"      # optional; defaults to "src/main.gcr"
//! ```
//!
//! We parse only `key = "value"` lines under a `[package]` section. This is a
//! strict subset of TOML — enough for v1, and it avoids pulling in a TOML crate
//! for what is a dozen lines of config.

use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Manifest {
    pub name: String,
    pub version: String,
    /// Entry source file, relative to the manifest directory.
    pub entry: PathBuf,
    /// The directory the manifest lives in.
    pub dir: PathBuf,
}

#[derive(Debug)]
pub struct ManifestError(pub String);

impl Manifest {
    /// The absolute path to the entry source file.
    pub fn entry_path(&self) -> PathBuf {
        self.dir.join(&self.entry)
    }

    /// Load `gcr.toml` from `dir`. Errors if the file is missing or malformed.
    pub fn load(dir: &Path) -> Result<Manifest, ManifestError> {
        let path = dir.join("gcr.toml");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| ManifestError(format!("cannot read {}: {}", path.display(), e)))?;
        Self::parse(&text, dir)
    }

    /// Find a `gcr.toml` by walking up from `start` to the filesystem root.
    /// Returns `None` if no manifest is found (a bare-file build is still valid).
    pub fn discover(start: &Path) -> Option<Manifest> {
        let mut dir = if start.is_dir() { start.to_path_buf() } else { start.parent()?.to_path_buf() };
        loop {
            if dir.join("gcr.toml").exists() {
                return Manifest::load(&dir).ok();
            }
            if !dir.pop() {
                return None;
            }
        }
    }

    fn parse(text: &str, dir: &Path) -> Result<Manifest, ManifestError> {
        let mut name: Option<String> = None;
        let mut version: Option<String> = None;
        let mut entry: Option<String> = None;
        let mut in_package = false;

        for (lineno, raw) in text.lines().enumerate() {
            let line = raw.split('#').next().unwrap().trim();
            if line.is_empty() {
                continue;
            }
            if line.starts_with('[') {
                in_package = line == "[package]";
                continue;
            }
            if !in_package {
                continue;
            }
            let Some((key, val)) = line.split_once('=') else {
                return Err(ManifestError(format!("gcr.toml:{}: expected `key = value`", lineno + 1)));
            };
            let key = key.trim();
            let val = val.trim().trim_matches('"').to_string();
            match key {
                "name" => name = Some(val),
                "version" => version = Some(val),
                "entry" => entry = Some(val),
                other => return Err(ManifestError(format!("gcr.toml:{}: unknown key `{}`", lineno + 1, other))),
            }
        }

        let name = name.ok_or_else(|| ManifestError("gcr.toml: missing `name`".into()))?;
        Ok(Manifest {
            name,
            version: version.unwrap_or_else(|| "0.0.0".into()),
            entry: PathBuf::from(entry.unwrap_or_else(|| "src/main.gcr".into())),
            dir: dir.to_path_buf(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_manifest() {
        let src = "[package]\nname = \"myapp\"\nversion = \"1.2.3\"\nentry = \"src/app.gcr\"\n";
        let m = Manifest::parse(src, Path::new("/proj")).unwrap();
        assert_eq!(m.name, "myapp");
        assert_eq!(m.version, "1.2.3");
        assert_eq!(m.entry_path(), Path::new("/proj/src/app.gcr"));
    }

    #[test]
    fn entry_defaults() {
        let src = "[package]\nname = \"x\"\n";
        let m = Manifest::parse(src, Path::new("/p")).unwrap();
        assert_eq!(m.version, "0.0.0");
        assert_eq!(m.entry, PathBuf::from("src/main.gcr"));
    }

    #[test]
    fn comments_and_blank_lines_ignored() {
        let src = "# a project\n\n[package]   # the package table\nname = \"c\"  # its name\n";
        let m = Manifest::parse(src, Path::new("/p")).unwrap();
        assert_eq!(m.name, "c");
    }

    #[test]
    fn missing_name_errors() {
        let src = "[package]\nversion = \"1.0.0\"\n";
        assert!(Manifest::parse(src, Path::new("/p")).is_err());
    }

    #[test]
    fn unknown_key_errors() {
        let src = "[package]\nname = \"x\"\nbogus = \"y\"\n";
        assert!(Manifest::parse(src, Path::new("/p")).is_err());
    }
}
