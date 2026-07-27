//! PostCSS integration.
//!
//! When a project carries a PostCSS config (`postcss.config.{js,cjs,mjs,ts}`,
//! `.postcssrc[.json]`, or a `package.json` `"postcss"` key), Diffpack runs the
//! app's OWN `postcss` and its configured plugins (autoprefixer, nesting, ...)
//! over every stylesheet — exactly as Vite does — before the native CSS
//! pipeline extracts `@import`s, rebases `url(...)`s, and scopes CSS Modules.
//! The compile is shelled to `node` (like [`crate::vite_config`] and the Sass
//! preprocessors), so the app's exact plugin toolchain and versions are used;
//! no PostCSS is reimplemented in Rust.
//!
//! Discovery walks from the project root upward to the filesystem root, taking
//! the first config found (postcss-load-config semantics). No config means no
//! PostCSS step at all — zero overhead beyond a handful of `stat`s at setup, so
//! a plain Vite/Tailwind app is unaffected.
//!
//! Results are cached by content within a build: the same CSS + `from` pair is
//! transformed once. A plugin the app configures but has not installed is a
//! hard, specific error (the runner names the missing package); output is never
//! silently passed through unprefixed.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;

/// The node script that loads the app's PostCSS config and runs its plugins.
const RUNNER: &str = include_str!("postcss_runner.mjs");

/// Where the discovered PostCSS config lives.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ConfigSource {
    /// A dedicated config file (`postcss.config.js`, `.postcssrc.json`, ...).
    File(PathBuf),
    /// The `"postcss"` key of a `package.json`.
    Package(PathBuf),
}

impl ConfigSource {
    fn path(&self) -> &Path {
        match self {
            ConfigSource::File(path) | ConfigSource::Package(path) => path,
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            ConfigSource::File(_) => "file",
            ConfigSource::Package(_) => "package",
        }
    }
}

/// A resolved PostCSS setup for one build: the discovered config, the project
/// root (the `node` working directory, so the app's `postcss`/plugins resolve),
/// and a per-build content cache.
#[derive(Debug)]
pub struct Postcss {
    source: ConfigSource,
    root: PathBuf,
    cache: Mutex<HashMap<u64, String>>,
}

/// Dedicated config filenames, in postcss-load-config's resolution order.
const CONFIG_FILES: [&str; 6] = [
    "postcss.config.js",
    "postcss.config.cjs",
    "postcss.config.mjs",
    "postcss.config.ts",
    ".postcssrc.json",
    ".postcssrc",
];

/// Locates the project's PostCSS config, walking from `root` upward. The walk
/// stops at the first `package.json` (the project root): a config only counts
/// when it belongs to this project, never a stray one in an unrelated monorepo
/// ancestor. Returns `None` when the project uses no PostCSS.
pub fn discover(root: &Path) -> Option<Postcss> {
    for directory in root.ancestors() {
        for name in CONFIG_FILES {
            let candidate = directory.join(name);
            if candidate.is_file() {
                return Some(Postcss::new(ConfigSource::File(candidate), root));
            }
        }
        let package = directory.join("package.json");
        if package.is_file() {
            // The project boundary: its `package.json` may carry the config,
            // and the walk goes no higher regardless.
            if package_has_postcss(&package) {
                return Some(Postcss::new(ConfigSource::Package(package), root));
            }
            return None;
        }
    }
    None
}

/// Whether a `package.json` declares a top-level `"postcss"` config object.
fn package_has_postcss(package: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(package) else {
        return false;
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    value
        .get("postcss")
        .is_some_and(|value| value.is_object() || value.is_array())
}

impl Postcss {
    fn new(source: ConfigSource, root: &Path) -> Self {
        Self {
            source,
            root: root.to_path_buf(),
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// The config file itself, so the caller can record it as a build input
    /// (editing the config re-derives every stylesheet).
    pub fn config_file(&self) -> &Path {
        self.source.path()
    }

    /// Transforms one stylesheet through the app's PostCSS. `from` is the
    /// stylesheet's own path (PostCSS `from`, used by file-relative plugins and
    /// diagnostics). Cached by `(css, from)` content within the build.
    pub fn process(&self, css: &str, from: &Path) -> Result<String, String> {
        let key = cache_key(css, from);
        if let Some(cached) = self.cache.lock().expect("postcss cache poisoned").get(&key) {
            return Ok(cached.clone());
        }
        let transformed = self.run_node(css, from)?;
        self.cache
            .lock()
            .expect("postcss cache poisoned")
            .insert(key, transformed.clone());
        Ok(transformed)
    }

    fn run_node(&self, css: &str, from: &Path) -> Result<String, String> {
        // The runner module is passed via `--eval`, freeing the child's stdin
        // to carry the CSS. (Piping the module on stdin — as the vite-config
        // evaluator does — would collide with the runner's own stdin read.)
        // Bare `import` specifiers in an `--eval` module resolve against the
        // working directory, so the project's own `postcss` and plugins load.
        let mut child = Command::new("node")
            .arg("--input-type=module")
            .arg("--no-warnings")
            .arg("--eval")
            .arg(RUNNER)
            .env("DIFFPACK_POSTCSS_CONFIG", self.source.path())
            .env("DIFFPACK_POSTCSS_CONFIG_KIND", self.source.kind())
            .env("DIFFPACK_POSTCSS_FROM", from)
            .current_dir(&self.root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| {
                format!(
                    "cannot spawn `node` to run PostCSS for {}: {error}",
                    from.display()
                )
            })?;

        {
            let mut stdin = child
                .stdin
                .take()
                .ok_or("failed to open node stdin for the PostCSS runner")?;
            stdin
                .write_all(css.as_bytes())
                .map_err(|error| format!("cannot pipe CSS to the PostCSS runner: {error}"))?;
        }

        let output = child
            .wait_with_output()
            .map_err(|error| format!("node (PostCSS runner) did not complete: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "PostCSS failed for {}: {}",
                from.display(),
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        String::from_utf8(output.stdout)
            .map_err(|error| format!("PostCSS produced non-UTF-8 CSS for {}: {error}", from.display()))
    }
}

/// A content key for the transform cache: the CSS plus the `from` path.
fn cache_key(css: &str, from: &Path) -> u64 {
    let mut hasher = DefaultHasher::new();
    css.hash(&mut hasher);
    0u8.hash(&mut hasher);
    from.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_dedicated_config_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("postcss.config.js"),
            "module.exports = { plugins: {} };\n",
        )
        .unwrap();
        let found = discover(dir.path()).expect("config discovered");
        assert!(matches!(found.source, ConfigSource::File(_)));
        assert_eq!(
            found.config_file().file_name().unwrap().to_str().unwrap(),
            "postcss.config.js"
        );
    }

    #[test]
    fn discovers_package_json_key() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("package.json"),
            "{ \"name\": \"x\", \"postcss\": { \"plugins\": {} } }",
        )
        .unwrap();
        let found = discover(dir.path()).expect("config discovered");
        assert!(matches!(found.source, ConfigSource::Package(_)));
    }

    #[test]
    fn no_config_means_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("package.json"), "{ \"name\": \"x\" }").unwrap();
        // Walking to the filesystem root must not find a stray config here.
        assert!(discover(dir.path()).is_none());
    }

    #[test]
    fn package_json_without_postcss_is_ignored() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("package.json"),
            "{ \"name\": \"x\", \"scripts\": {} }",
        )
        .unwrap();
        assert!(!package_has_postcss(&dir.path().join("package.json")));
    }

    #[test]
    fn cache_key_is_sensitive_to_css_and_from() {
        let a = cache_key(".x{}", Path::new("/a.css"));
        let b = cache_key(".y{}", Path::new("/a.css"));
        let c = cache_key(".x{}", Path::new("/b.css"));
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_eq!(a, cache_key(".x{}", Path::new("/a.css")));
    }
}
