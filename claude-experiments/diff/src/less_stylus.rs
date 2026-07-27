//! Less and Stylus preprocessing.
//!
//! A `.less` or `.styl` source is compiled to plain CSS *before* the shared CSS
//! pipeline runs, so the result flows through the same CSS-Modules / `@import` /
//! `url()` rebasing stages as a hand-written `.css` file — the identical shape
//! as [`crate::sass`]. Unlike Sass (reimplemented natively), Less and Stylus are
//! compiled by shelling to the APP's OWN `less` / `stylus` package (`node` run
//! with cwd = project root), so the project's exact preprocessor and its plugins
//! are used and no preprocessor is reimplemented in Rust.
//!
//! The compiler reports the other files it pulled in via `@import` (`deps`);
//! those are recorded by the caller so editing a partial re-derives the module.
//! A missing `less`/`stylus` package is a hard, specific error (the runner names
//! it); the source is never passed through unprocessed.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The node script that compiles Less/Stylus with the app's own toolchain.
const RUNNER: &str = include_str!("less_stylus_runner.mjs");

/// Which preprocessor a source uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tool {
    Less,
    Stylus,
}

impl Tool {
    fn env_value(self) -> &'static str {
        match self {
            Tool::Less => "less",
            Tool::Stylus => "stylus",
        }
    }

    fn name(self) -> &'static str {
        self.env_value()
    }
}

/// The result of compiling one Less/Stylus root file.
#[derive(Debug, Clone)]
pub struct CompiledCss {
    /// Plain CSS, ready for [`crate::css::process_global_css`] or
    /// [`crate::css::process_css_module`].
    pub css: String,
    /// Every OTHER file the compile pulled in (`@import` targets), so edits to
    /// any of them invalidate the owning module.
    pub loaded_files: Vec<PathBuf>,
}

/// Whether a resolved path is a Less source (`.less`).
pub fn is_less_path(path: &Path) -> bool {
    path.extension().and_then(|value| value.to_str()) == Some("less")
}

/// Whether a resolved path is a Stylus source (`.styl` or `.stylus`).
pub fn is_stylus_path(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|value| value.to_str()),
        Some("styl" | "stylus")
    )
}

/// Whether a resolved path is any Less/Stylus source.
pub fn is_less_or_stylus_path(path: &Path) -> bool {
    is_less_path(path) || is_stylus_path(path)
}

/// Whether a resolved path is a Less CSS Module (`*.module.less`).
pub fn is_less_module_path(path: &Path) -> bool {
    has_module_suffix(path, ".module.less")
}

/// Whether a resolved path is a Stylus CSS Module (`*.module.styl(us)`).
pub fn is_stylus_module_path(path: &Path) -> bool {
    has_module_suffix(path, ".module.styl") || has_module_suffix(path, ".module.stylus")
}

/// Whether a resolved path is any Less/Stylus CSS Module.
pub fn is_css_module_path(path: &Path) -> bool {
    is_less_module_path(path) || is_stylus_module_path(path)
}

fn has_module_suffix(path: &Path, suffix: &str) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(suffix) && name.len() > suffix.len())
}

/// Compiles one `.less` file to plain CSS with the app's `less`. `root`, when
/// set, is the `node` working directory (so `import 'less'` resolves the
/// project's copy); otherwise the source file's directory is used.
pub fn compile_less(file: &Path, source: &str, root: Option<&Path>) -> Result<CompiledCss, String> {
    compile(Tool::Less, file, source, root)
}

/// Compiles one `.styl` file to plain CSS with the app's `stylus`.
pub fn compile_stylus(file: &Path, source: &str, root: Option<&Path>) -> Result<CompiledCss, String> {
    compile(Tool::Stylus, file, source, root)
}

fn compile(tool: Tool, file: &Path, source: &str, root: Option<&Path>) -> Result<CompiledCss, String> {
    let working_dir = root
        .map(Path::to_path_buf)
        .or_else(|| file.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."));

    // The runner module is passed via `--eval` so the child's stdin carries the
    // source; bare `import` specifiers resolve against `working_dir`.
    let mut child = Command::new("node")
        .arg("--input-type=module")
        .arg("--no-warnings")
        .arg("--eval")
        .arg(RUNNER)
        .env("DIFFPACK_CSS_TOOL", tool.env_value())
        .env("DIFFPACK_CSS_FILE", file)
        .current_dir(&working_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            format!(
                "cannot spawn `node` to compile {} ({}): {error}",
                file.display(),
                tool.name()
            )
        })?;

    {
        let mut stdin = child
            .stdin
            .take()
            .ok_or("failed to open node stdin for the Less/Stylus runner")?;
        stdin
            .write_all(source.as_bytes())
            .map_err(|error| format!("cannot pipe source to the {} runner: {error}", tool.name()))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|error| format!("node ({} runner) did not complete: {error}", tool.name()))?;
    if !output.status.success() {
        return Err(format!(
            "{} failed for {}: {}",
            tool.name(),
            file.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    parse_output(tool, file, &output.stdout)
}

/// Parses the runner's `{ "css": ..., "deps": [..] }` JSON. `deps` are absolute
/// paths of imported files; the source file itself is filtered out.
fn parse_output(tool: Tool, file: &Path, stdout: &[u8]) -> Result<CompiledCss, String> {
    let value: serde_json::Value = serde_json::from_slice(stdout).map_err(|error| {
        format!(
            "{} produced invalid JSON for {}: {error}",
            tool.name(),
            file.display()
        )
    })?;
    let css = value
        .get("css")
        .and_then(|css| css.as_str())
        .ok_or_else(|| format!("{} output for {} had no \"css\"", tool.name(), file.display()))?
        .to_string();
    let mut loaded_files = Vec::new();
    if let Some(serde_json::Value::Array(deps)) = value.get("deps") {
        let canonical_self = file.canonicalize().ok();
        for dep in deps {
            if let Some(dep) = dep.as_str() {
                let path = PathBuf::from(dep);
                // The source itself is not a *dependency*. Match on the raw path,
                // or on canonical paths only when BOTH resolve (a failed
                // canonicalize must not collapse distinct files to "equal").
                let is_self = path == file
                    || matches!(
                        (&canonical_self, path.canonicalize().ok()),
                        (Some(a), Some(b)) if *a == b
                    );
                if !is_self && !loaded_files.contains(&path) {
                    loaded_files.push(path);
                }
            }
        }
    }
    Ok(CompiledCss { css, loaded_files })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn less_extension_detection() {
        assert!(is_less_path(Path::new("a/b.less")));
        assert!(!is_less_path(Path::new("a/b.css")));
        assert!(is_less_module_path(Path::new("a/x.module.less")));
        assert!(!is_less_module_path(Path::new("a/x.less")));
        assert!(!is_less_module_path(Path::new("a/.module.less")));
    }

    #[test]
    fn stylus_extension_detection() {
        assert!(is_stylus_path(Path::new("a/b.styl")));
        assert!(is_stylus_path(Path::new("a/b.stylus")));
        assert!(!is_stylus_path(Path::new("a/b.less")));
        assert!(is_stylus_module_path(Path::new("a/x.module.styl")));
        assert!(is_stylus_module_path(Path::new("a/x.module.stylus")));
        assert!(!is_stylus_module_path(Path::new("a/x.styl")));
    }

    #[test]
    fn combined_predicates() {
        assert!(is_less_or_stylus_path(Path::new("a.less")));
        assert!(is_less_or_stylus_path(Path::new("a.styl")));
        assert!(!is_less_or_stylus_path(Path::new("a.scss")));
        assert!(is_css_module_path(Path::new("a.module.less")));
        assert!(is_css_module_path(Path::new("a.module.styl")));
        assert!(!is_css_module_path(Path::new("a.less")));
    }

    #[test]
    fn parses_runner_json_and_filters_self() {
        let file = Path::new("/proj/x.less");
        let json = br#"{"css":".a{color:red}","deps":["/proj/x.less","/proj/_vars.less"]}"#;
        let compiled = parse_output(Tool::Less, file, json).expect("parse");
        assert_eq!(compiled.css, ".a{color:red}");
        assert_eq!(compiled.loaded_files, vec![PathBuf::from("/proj/_vars.less")]);
    }

    #[test]
    fn missing_css_field_is_an_error() {
        let file = Path::new("/proj/x.styl");
        let json = br#"{"deps":[]}"#;
        assert!(parse_output(Tool::Stylus, file, json).is_err());
    }
}
