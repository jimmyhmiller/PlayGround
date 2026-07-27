//! Vue and Svelte single-file components.
//!
//! A `.vue` or `.svelte` file is not JavaScript: its `<template>`/markup is a
//! separate language that only the framework's own compiler understands. Both
//! compilers ship as ordinary npm packages that every such app already depends
//! on (`@vue/compiler-sfc` via `@vitejs/plugin-vue`, `svelte/compiler` via
//! `@sveltejs/vite-plugin-svelte`), so a component is compiled by shelling to
//! the APP's OWN copy (`node` run with cwd = project root) — the identical shape
//! as [`crate::less_stylus`], which compiles Less/Stylus with the app's own
//! preprocessor. diffpack reimplements neither compiler and hosts no JS plugin.
//!
//! The compiler's JavaScript output is deliberately NOT final. It still carries
//! the component's own `import`s (the framework runtime, sibling components,
//! asset URLs) and, for a `<script lang="ts">` Vue SFC, TypeScript annotations.
//! The caller feeds it through the ordinary module pipeline so those imports
//! become real graph edges and the types are stripped by the same transform a
//! `.ts` module takes — which is exactly what `@vitejs/plugin-vue` does with its
//! own output (`transformWithOxc(..., { lang: "ts" })`). Style blocks come back
//! separately as plain CSS and join the build's stylesheet.
//!
//! A missing or broken compiler is a hard, specific error naming the file and
//! the package; a component is never passed through as JavaScript.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The node script that compiles a component with the app's own toolchain.
const RUNNER: &str = include_str!("sfc_runner.mjs");

/// A component format diffpack compiles through the app's own compiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Framework {
    Vue,
    Svelte,
}

impl Framework {
    /// The `DIFFPACK_SFC_FRAMEWORK` value the runner dispatches on.
    fn env_value(self) -> &'static str {
        match self {
            Framework::Vue => "vue",
            Framework::Svelte => "svelte",
        }
    }

    /// How the format reads in an error message.
    fn describe(self) -> &'static str {
        match self {
            Framework::Vue => "Vue single-file component",
            Framework::Svelte => "Svelte component",
        }
    }
}

/// Which language the compiled JavaScript is written in. Vue's SFC compiler
/// leaves a `<script lang="ts">` component's annotations in place (plugin-vue
/// hands them to Vite's TypeScript transform afterwards); Svelte's compiler
/// strips them itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputLanguage {
    JavaScript,
    TypeScript,
}

/// One compiled component.
#[derive(Debug, Clone)]
pub struct CompiledComponent {
    /// The component's JavaScript. Still contains `import`s and, when
    /// [`Self::language`] is TypeScript, type annotations.
    pub code: String,
    /// The component's style blocks as plain CSS (`None` when it has none).
    /// Already scoped by the framework's compiler when the block asked for it.
    pub css: Option<String>,
    /// How [`Self::code`] must be parsed.
    pub language: OutputLanguage,
}

/// The component format `path` is, or `None` for anything else (including
/// ordinary JavaScript and TypeScript).
pub fn framework_for(path: &Path) -> Option<Framework> {
    match path.extension().and_then(|value| value.to_str()) {
        Some("vue") => Some(Framework::Vue),
        Some("svelte") => Some(Framework::Svelte),
        _ => None,
    }
}

/// Whether `path` is a single-file component of any supported framework.
pub fn is_component_path(path: &Path) -> bool {
    framework_for(path).is_some()
}

/// Compiles one component with the app's own compiler. `root`, when set, is the
/// `node` working directory (so `import "@vue/compiler-sfc"` resolves the
/// project's copy) and the base the Vue component id hash is taken against;
/// otherwise the component's own directory is used.
pub fn compile(
    framework: Framework,
    file: &Path,
    source: &str,
    root: Option<&Path>,
) -> Result<CompiledComponent, String> {
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
        .env("DIFFPACK_SFC_FRAMEWORK", framework.env_value())
        .env("DIFFPACK_SFC_FILE", file)
        .env("DIFFPACK_SFC_ROOT", &working_dir)
        .current_dir(&working_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            format!(
                "cannot spawn `node` to compile the {} {} : {error}",
                framework.describe(),
                file.display()
            )
        })?;

    {
        let mut stdin = child
            .stdin
            .take()
            .ok_or("failed to open node stdin for the single-file-component runner")?;
        stdin.write_all(source.as_bytes()).map_err(|error| {
            format!(
                "cannot pipe {} to the {} compiler: {error}",
                file.display(),
                framework.describe()
            )
        })?;
    }

    let output = child.wait_with_output().map_err(|error| {
        format!(
            "node (the {} compiler runner) did not complete: {error}",
            framework.describe()
        )
    })?;
    if !output.status.success() {
        return Err(format!(
            "cannot compile the {} {}: {}",
            framework.describe(),
            file.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    parse_output(framework, file, &output.stdout)
}

/// Parses the runner's `{ "code": ..., "css": ..., "language": ... }` JSON.
/// Every field is required: a missing one means the runner and this module
/// disagree, which must fail loudly rather than compile the component to
/// nothing.
fn parse_output(
    framework: Framework,
    file: &Path,
    stdout: &[u8],
) -> Result<CompiledComponent, String> {
    let value: serde_json::Value = serde_json::from_slice(stdout).map_err(|error| {
        format!(
            "the {} compiler produced invalid JSON for {}: {error}",
            framework.describe(),
            file.display()
        )
    })?;
    let code = value
        .get("code")
        .and_then(|code| code.as_str())
        .ok_or_else(|| {
            format!(
                "the {} compiler's output for {} had no \"code\"",
                framework.describe(),
                file.display()
            )
        })?
        .to_string();
    let css = value
        .get("css")
        .and_then(|css| css.as_str())
        .filter(|css| !css.trim().is_empty())
        .map(str::to_string);
    let language = match value.get("language").and_then(|value| value.as_str()) {
        Some("ts") => OutputLanguage::TypeScript,
        Some("js") => OutputLanguage::JavaScript,
        other => {
            return Err(format!(
                "the {} compiler's output for {} had an unusable \"language\" ({:?}); \
                 expected \"ts\" or \"js\"",
                framework.describe(),
                file.display(),
                other
            ));
        }
    };
    Ok(CompiledComponent {
        code,
        css,
        language,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_extension_detection() {
        assert_eq!(framework_for(Path::new("a/App.vue")), Some(Framework::Vue));
        assert_eq!(
            framework_for(Path::new("a/App.svelte")),
            Some(Framework::Svelte)
        );
        assert_eq!(framework_for(Path::new("a/App.tsx")), None);
        assert_eq!(framework_for(Path::new("a/App")), None);
        assert!(is_component_path(Path::new("x.vue")));
        assert!(!is_component_path(Path::new("x.ts")));
    }

    #[test]
    fn parses_runner_json() {
        let json = br#"{"code":"export default {}","css":".a{color:red}","language":"ts"}"#;
        let compiled = parse_output(Framework::Vue, Path::new("/p/A.vue"), json).expect("parse");
        assert_eq!(compiled.code, "export default {}");
        assert_eq!(compiled.css.as_deref(), Some(".a{color:red}"));
        assert_eq!(compiled.language, OutputLanguage::TypeScript);
    }

    #[test]
    fn blank_css_is_no_css() {
        let json = br#"{"code":"x","css":"  \n ","language":"js"}"#;
        let compiled = parse_output(Framework::Svelte, Path::new("/p/A.svelte"), json).expect("parse");
        assert_eq!(compiled.css, None);
        assert_eq!(compiled.language, OutputLanguage::JavaScript);
    }

    #[test]
    fn a_missing_language_is_an_error() {
        let json = br#"{"code":"x","css":""}"#;
        let error =
            parse_output(Framework::Vue, Path::new("/p/A.vue"), json).expect_err("must reject");
        assert!(error.contains("language"), "{error}");
        assert!(error.contains("A.vue"), "{error}");
    }

    #[test]
    fn a_missing_code_field_is_an_error() {
        let json = br#"{"css":"","language":"js"}"#;
        let error =
            parse_output(Framework::Svelte, Path::new("/p/A.svelte"), json).expect_err("must reject");
        assert!(error.contains("code"), "{error}");
    }
}
