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
    /// Native link configuration (the `[link]` section). Empty by default.
    pub link: LinkConfig,
}

/// The `[link]` section of `gcr.toml` — how to link native libraries into the
/// final executable, so `gcr build`/`gcr run` need no `--link-arg` on the CLI.
///
/// ```toml
/// [link]
/// libs = ["raylib"]                       # -l<name>
/// lib-paths = ["/opt/homebrew/lib"]        # -L<path>
/// frameworks = ["Cocoa", "OpenGL"]         # macOS: -framework <name>
/// args = ["-Wl,-rpath,/opt/homebrew/lib"]  # raw, passed through verbatim
/// ```
#[derive(Debug, Clone, Default)]
pub struct LinkConfig {
    pub libs: Vec<String>,
    pub lib_paths: Vec<String>,
    pub frameworks: Vec<String>,
    pub args: Vec<String>,
}

impl LinkConfig {
    /// Flatten into the linker-argument list `build_executable` expects.
    pub fn to_args(&self) -> Vec<String> {
        let mut out = Vec::new();
        for p in &self.lib_paths {
            out.push(format!("-L{}", p));
        }
        for l in &self.libs {
            out.push(format!("-l{}", l));
        }
        for f in &self.frameworks {
            out.push("-framework".into());
            out.push(f.clone());
        }
        out.extend(self.args.iter().cloned());
        out
    }
}

#[derive(Debug)]
pub struct ManifestError(pub String);

impl Manifest {
    /// The absolute path to the entry source file.
    pub fn entry_path(&self) -> PathBuf {
        self.dir.join(&self.entry)
    }

    /// Native linker arguments from the `[link]` section.
    pub fn link_args(&self) -> Vec<String> {
        self.link.to_args()
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
        let mut link = LinkConfig::default();
        let mut section = "";

        for (lineno, raw) in text.lines().enumerate() {
            let line = raw.split('#').next().unwrap().trim();
            if line.is_empty() {
                continue;
            }
            if line.starts_with('[') {
                section = match line {
                    "[package]" => "package",
                    "[link]" => "link",
                    other => {
                        return Err(ManifestError(format!(
                            "gcr.toml:{}: unknown section `{}`",
                            lineno + 1,
                            other
                        )))
                    }
                };
                continue;
            }
            let Some((key, val)) = line.split_once('=') else {
                return Err(ManifestError(format!("gcr.toml:{}: expected `key = value`", lineno + 1)));
            };
            let key = key.trim();
            let val = val.trim();
            match (section, key) {
                ("package", "name") => name = Some(unquote(val)),
                ("package", "version") => version = Some(unquote(val)),
                ("package", "entry") => entry = Some(unquote(val)),
                ("link", "libs") => link.libs = parse_array(val),
                ("link", "lib-paths") => link.lib_paths = parse_array(val),
                ("link", "frameworks") => link.frameworks = parse_array(val),
                ("link", "args") => link.args = parse_array(val),
                (sec, other) => {
                    return Err(ManifestError(format!(
                        "gcr.toml:{}: unknown key `{}` in [{}]",
                        lineno + 1,
                        other,
                        if sec.is_empty() { "<top-level>" } else { sec }
                    )))
                }
            }
        }

        let name = name.ok_or_else(|| ManifestError("gcr.toml: missing `name`".into()))?;
        Ok(Manifest {
            name,
            version: version.unwrap_or_else(|| "0.0.0".into()),
            entry: PathBuf::from(entry.unwrap_or_else(|| "src/main.gcr".into())),
            dir: dir.to_path_buf(),
            link,
        })
    }
}

/// Strip surrounding double quotes (and whitespace) from a scalar TOML value.
fn unquote(val: &str) -> String {
    val.trim().trim_matches('"').to_string()
}

/// Parse a TOML string array `["a", "b"]` into its elements. Splits on commas
/// OUTSIDE double quotes, so an element may itself contain commas (e.g. a raw
/// linker arg `"-Wl,-rpath,/x"`). Lenient about trailing commas + whitespace.
/// (Single-line arrays only — enough for link config.)
fn parse_array(val: &str) -> Vec<String> {
    let inner = val.trim().trim_start_matches('[').trim_end_matches(']');
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_quote = false;
    for c in inner.chars() {
        match c {
            '"' => in_quote = !in_quote,
            ',' if !in_quote => {
                let e = cur.trim().to_string();
                if !e.is_empty() {
                    out.push(e);
                }
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    let e = cur.trim().to_string();
    if !e.is_empty() {
        out.push(e);
    }
    out
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
    fn parses_link_section_into_linker_args() {
        let src = "[package]\nname = \"game\"\n\n[link]\n\
                   libs = [\"raylib\"]\n\
                   lib-paths = [\"/opt/homebrew/lib\"]\n\
                   frameworks = [\"Cocoa\", \"OpenGL\"]\n\
                   args = [\"-Wl,-rpath,/x\"]\n";
        let m = Manifest::parse(src, Path::new("/p")).unwrap();
        assert_eq!(m.link.libs, ["raylib"]);
        assert_eq!(
            m.link_args(),
            [
                "-L/opt/homebrew/lib",
                "-lraylib",
                "-framework",
                "Cocoa",
                "-framework",
                "OpenGL",
                "-Wl,-rpath,/x",
            ]
        );
    }

    #[test]
    fn no_link_section_means_no_args() {
        let m = Manifest::parse("[package]\nname = \"x\"\n", Path::new("/p")).unwrap();
        assert!(m.link_args().is_empty());
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
