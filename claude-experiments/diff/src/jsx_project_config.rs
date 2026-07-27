//! Which `tsconfig.json` / `jsconfig.json` configures a file's JSX lowering.
//!
//! This is deliberately NOT the resolver's `find_tsconfig`. That answers
//! TypeScript's question — *would `tsc` type-check this file as part of this
//! project?* — and answering it decides two things at once that must not be
//! decided together:
//!
//! * `include: ["src"]` claims `src/app.tsx` but not `src/app.jsx`, because
//!   TypeScript will not compile a `.js`-family file without `allowJs`. The
//!   `jsxImportSource` in that config is nevertheless the only statement the
//!   project ever makes about which package its JSX comes from, and a bundler
//!   lowers the `.jsx` file whatever `tsc` would have done with it.
//! * `include: ["**/*.ts", "**/*.tsx"]` is what `create-next-app` writes, and
//!   Next compiles JSX in `.js` too. Under the type-checking rule one app gets
//!   two JSX runtimes: `.tsx` modules on the configured import source and `.js`
//!   modules silently on React.
//!
//! The rule here is therefore **applicability, not compilation**: the nearest
//! config whose `files`/`include`/`exclude` would cover the file *if its
//! extension were any member of the JS/TS family* owns the file's JSX. That is
//! what every one of these toolchains actually does — Vite hands the whole
//! transform one JSX contract per project, and Next's SWC loader reads
//! `jsx`/`jsxImportSource` from `tsconfig.json` **or `jsconfig.json`** and applies
//! it to every module it compiles.
//!
//! Everything else about the config (`paths`, `baseUrl`, references, `extends`)
//! is still the resolver's; this module only decides WHICH config, and loads it
//! through the resolver so `extends` and project references resolve identically.

use std::path::Path;
use std::sync::Arc;

use oxc_resolver::{Resolver, TsConfig};

/// Where a JS/TS project states its compiler options, in the order TypeScript
/// looks within one directory. `jsconfig.json` is TypeScript's own spelling for a
/// JavaScript project — for a project with no `.ts` at all it is the ONLY place
/// `jsx`/`jsxImportSource` can be written, and both Next and the TS language
/// service read it.
const CONFIG_FILE_NAMES: [&str; 2] = ["tsconfig.json", "jsconfig.json"];

/// The extensions this bundler lowers JSX from, treated as ONE family when asking
/// whether a config applies. `.md`/`.mdx` are here because MDX compiles to JSX and
/// is then transformed exactly like a `.tsx` module (`crate::transform`), so it
/// needs the same import source as the components it renders.
const PROGRAM_EXTENSIONS: [&str; 10] = [
    "ts", "tsx", "mts", "cts", "js", "jsx", "mjs", "cjs", "mdx", "md",
];

/// The `tsconfig.json`/`jsconfig.json` whose `compilerOptions` configure `path`'s
/// JSX, or `None` when no config in any ancestor directory applies to it (which
/// leaves oxc's default: the automatic runtime against `react`).
///
/// Walks up from the file's directory. Within a directory, `tsconfig.json` is
/// preferred over `jsconfig.json` (TypeScript's own precedence). A config that
/// could not apply to the file is skipped and the walk continues, so a nested
/// `tsconfig.json` covering only `src` does not capture a sibling `scripts/`
/// file that an outer config does cover.
///
/// # Errors
///
/// When a discovered config (or one it `extends`/`references`) cannot be read or
/// parsed. The message names the config file.
pub fn owning_config(resolver: &Resolver, path: &Path) -> Result<Option<Arc<TsConfig>>, String> {
    // A dependency's own sources are compiled with the dependency's contract, not
    // the app's — the same boundary `find_tsconfig` draws.
    if !path.is_absolute() || is_inside_node_modules(path) {
        return Ok(None);
    }
    let mut directory = path.parent();
    while let Some(current) = directory {
        for name in CONFIG_FILE_NAMES {
            let config_path = current.join(name);
            if !config_path.is_file() {
                continue;
            }
            let config = resolver.resolve_tsconfig(&config_path).map_err(|error| {
                format!("cannot read {} (it configures how {}'s JSX is lowered): {error}",
                    config_path.display(),
                    path.display())
            })?;
            if let Some(applicable) = applicable_project(&config, path) {
                return Ok(Some(applicable));
            }
        }
        directory = current.parent();
    }
    Ok(None)
}

fn is_inside_node_modules(path: &Path) -> bool {
    path.components()
        .any(|component| component.as_os_str() == "node_modules")
}

/// The project within `config` that applies to `path`: a referenced sub-project
/// that covers it (create-vite's root config is `{"files":[],"references":[...]}`
/// with no `compilerOptions` at all, and the real settings live in
/// `tsconfig.app.json`), else `config` itself when it covers the file.
fn applicable_project(config: &Arc<TsConfig>, path: &Path) -> Option<Arc<TsConfig>> {
    if let Some(referenced) = config
        .references_resolved
        .iter()
        .find(|referenced| covers(referenced, path))
    {
        return Some(Arc::clone(referenced));
    }
    // A solution-style config — references plus an EXPLICITLY empty `files` and
    // `include` — states that it owns no files itself. An omitted `include`
    // defaults to `**/*` and must not be read this way.
    let is_solution_style = !config.references_resolved.is_empty()
        && matches!(config.files.as_deref(), Some([]))
        && matches!(config.include.as_deref(), Some([]));
    if is_solution_style {
        return None;
    }
    covers(config, path).then(|| Arc::clone(config))
}

/// Whether `config`'s file set would cover `path` if `path`'s extension were any
/// member of the JS/TS family — the applicability rule this module exists for.
fn covers(config: &TsConfig, path: &Path) -> bool {
    let Some(candidates) = family_candidates(path) else {
        // Not something whose JSX this bundler lowers (`.css`, `.vue`, `.svelte`,
        // an image): no JSX contract to inherit.
        return false;
    };
    if let Some(files) = &config.files
        && files.iter().any(|file| matches_any(file, &candidates))
    {
        return true;
    }
    let included = match &config.include {
        Some(patterns) => patterns
            .iter()
            .any(|pattern| pattern_covers(pattern, &candidates)),
        // No `include`: TypeScript's default is `**/*` under the config's own
        // directory — unless `files` is present, which then IS the whole file set.
        None => config.files.is_none() && path.starts_with(config.directory()),
    };
    if !included {
        return false;
    }
    !config
        .exclude
        .as_ref()
        .is_some_and(|patterns| {
            patterns
                .iter()
                .any(|pattern| pattern_covers(pattern, &candidates))
        })
}

/// `path` rewritten with every JS/TS-family extension, forward-slashed for glob
/// matching. `None` when `path` is not a file this bundler lowers JSX from.
fn family_candidates(path: &Path) -> Option<Vec<String>> {
    let extension = path.extension().and_then(|value| value.to_str())?;
    if !PROGRAM_EXTENSIONS.contains(&extension) {
        return None;
    }
    let mut candidates = Vec::with_capacity(PROGRAM_EXTENSIONS.len());
    candidates.push(forward_slashes(&path.to_string_lossy()));
    for candidate_extension in PROGRAM_EXTENSIONS {
        if candidate_extension == extension {
            continue;
        }
        candidates.push(forward_slashes(
            &path.with_extension(candidate_extension).to_string_lossy(),
        ));
    }
    Some(candidates)
}

fn forward_slashes(path: &str) -> String {
    if path.contains('\\') {
        path.replace('\\', "/")
    } else {
        path.to_string()
    }
}

/// A `files` entry (an exact path, already absolute) against the candidates.
fn matches_any(file: &Path, candidates: &[String]) -> bool {
    let file = forward_slashes(&file.to_string_lossy());
    candidates.contains(&file)
}

/// One `include`/`exclude` pattern (absolute, as the resolver normalizes them)
/// against the candidates, with TypeScript's implicit `/**/*` for a pattern that
/// names a directory.
fn pattern_covers(pattern: &Path, candidates: &[String]) -> bool {
    let pattern = forward_slashes(&pattern.to_string_lossy());
    if candidates.contains(&pattern) {
        return true;
    }
    let last_segment = pattern.rsplit('/').next().unwrap_or(pattern.as_str());
    let pattern = if last_segment.contains(['.', '*', '?']) {
        pattern
    } else if pattern.ends_with('/') {
        format!("{pattern}**/*")
    } else {
        format!("{pattern}/**/*")
    };
    candidates
        .iter()
        .any(|candidate| fast_glob::glob_match(pattern.as_str(), candidate.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxc_resolver::{ResolveOptions, TsconfigDiscovery};
    use std::fs;
    use tempfile::tempdir;

    fn resolver() -> Resolver {
        Resolver::new(ResolveOptions {
            tsconfig: Some(TsconfigDiscovery::Auto),
            ..ResolveOptions::default()
        })
    }

    fn import_source(resolver: &Resolver, path: &Path) -> Option<String> {
        owning_config(resolver, path)
            .unwrap()
            .and_then(|config| config.compiler_options.jsx_import_source.clone())
    }

    /// TypeScript's `include: ["src"]` does not claim a `.jsx` file without
    /// `allowJs` — but the `jsxImportSource` beside it is still the only statement
    /// the project makes about which package its JSX comes from.
    #[test]
    fn a_jsx_file_is_covered_by_a_tsconfig_that_would_not_type_check_it() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("tsconfig.json"),
            r#"{"compilerOptions":{"jsx":"react-jsx","jsxImportSource":"preact"},"include":["src"]}"#,
        )
        .unwrap();
        let resolver = resolver();
        assert_eq!(
            import_source(&resolver, &root.join("src").join("app.jsx")).as_deref(),
            Some("preact")
        );
        assert_eq!(
            import_source(&resolver, &root.join("src").join("app.tsx")).as_deref(),
            Some("preact")
        );
    }

    /// A JavaScript project puts its compiler options in `jsconfig.json`; nothing
    /// else in the project can carry `jsxImportSource`.
    #[test]
    fn a_jsconfig_configures_jsx_for_a_javascript_project() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("jsconfig.json"),
            r#"{"compilerOptions":{"jsx":"react-jsx","jsxImportSource":"preact"}}"#,
        )
        .unwrap();
        let resolver = resolver();
        assert_eq!(
            import_source(&resolver, &root.join("src").join("main.jsx")).as_deref(),
            Some("preact")
        );
    }

    /// `tsconfig.json` wins over a `jsconfig.json` in the same directory, as it
    /// does for TypeScript itself.
    #[test]
    fn a_tsconfig_outranks_a_jsconfig_in_the_same_directory() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::write(
            root.join("tsconfig.json"),
            r#"{"compilerOptions":{"jsxImportSource":"from-tsconfig"}}"#,
        )
        .unwrap();
        fs::write(
            root.join("jsconfig.json"),
            r#"{"compilerOptions":{"jsxImportSource":"from-jsconfig"}}"#,
        )
        .unwrap();
        assert_eq!(
            import_source(&resolver(), &root.join("app.jsx")).as_deref(),
            Some("from-tsconfig")
        );
    }

    /// `create-next-app`'s own `include` names only `.ts`/`.tsx`, and Next compiles
    /// JSX in `.js` too. Both extensions must land on ONE runtime.
    #[test]
    fn a_next_style_typescript_include_still_covers_the_projects_js_files() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::create_dir_all(root.join("components")).unwrap();
        fs::write(
            root.join("tsconfig.json"),
            r#"{"compilerOptions":{"jsx":"preserve","jsxImportSource":"preact"},
                "include":["next-env.d.ts","**/*.ts","**/*.tsx",".next/types/**/*.ts"],
                "exclude":["node_modules"]}"#,
        )
        .unwrap();
        let resolver = resolver();
        for module in ["components/card.js", "components/card.tsx", "app/page.mdx"] {
            assert_eq!(
                import_source(&resolver, &root.join(module)).as_deref(),
                Some("preact"),
                "{module} must get the app's single JSX runtime"
            );
        }
    }

    /// The solution-style root config create-vite writes carries no
    /// `compilerOptions` at all: the referenced sub-project that covers the file is
    /// the one that configures it.
    #[test]
    fn a_solution_style_root_defers_to_the_reference_that_covers_the_file() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("tsconfig.json"),
            r#"{"files":[],"references":[{"path":"./tsconfig.app.json"},{"path":"./tsconfig.node.json"}]}"#,
        )
        .unwrap();
        fs::write(
            root.join("tsconfig.app.json"),
            r#"{"compilerOptions":{"jsx":"react-jsx","jsxImportSource":"preact"},"include":["src"]}"#,
        )
        .unwrap();
        fs::write(
            root.join("tsconfig.node.json"),
            r#"{"compilerOptions":{"jsxImportSource":"node-only"},"include":["vite.config.ts"]}"#,
        )
        .unwrap();
        let resolver = resolver();
        assert_eq!(
            import_source(&resolver, &root.join("src").join("app.jsx")).as_deref(),
            Some("preact")
        );
        assert_eq!(
            import_source(&resolver, &root.join("vite.config.ts")).as_deref(),
            Some("node-only")
        );
    }

    /// A config that cannot apply to the file is skipped, not applied: the walk
    /// continues to an ancestor that does cover it.
    #[test]
    fn a_config_that_excludes_the_file_defers_to_an_ancestor() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::create_dir_all(root.join("packages").join("app").join("legacy")).unwrap();
        fs::write(
            root.join("tsconfig.json"),
            r#"{"compilerOptions":{"jsxImportSource":"outer"}}"#,
        )
        .unwrap();
        fs::write(
            root.join("packages").join("app").join("tsconfig.json"),
            r#"{"compilerOptions":{"jsxImportSource":"inner"},"exclude":["legacy"]}"#,
        )
        .unwrap();
        let resolver = resolver();
        assert_eq!(
            import_source(&resolver, &root.join("packages/app/src/app.jsx")).as_deref(),
            Some("inner")
        );
        assert_eq!(
            import_source(&resolver, &root.join("packages/app/legacy/old.jsx")).as_deref(),
            Some("outer")
        );
    }

    /// A dependency's sources keep the dependency's own contract; an app-level
    /// import source must never reach them.
    #[test]
    fn a_dependency_under_node_modules_gets_no_project_config() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::write(
            root.join("tsconfig.json"),
            r#"{"compilerOptions":{"jsxImportSource":"preact"}}"#,
        )
        .unwrap();
        assert!(
            owning_config(
                &resolver(),
                &root.join("node_modules").join("dep").join("index.jsx")
            )
            .unwrap()
            .is_none()
        );
    }

    /// A file with no config anywhere above it has no project contract, which is
    /// what leaves oxc's react default in charge.
    #[test]
    fn a_file_with_no_config_above_it_has_no_owning_project() {
        let directory = tempdir().unwrap();
        assert!(
            owning_config(&resolver(), &directory.path().join("src").join("app.jsx"))
                .unwrap()
                .is_none()
        );
    }

    /// A malformed config is a named hard error, never a silent fall-through to
    /// React (which would lower the whole project against a package it does not
    /// depend on).
    #[test]
    fn an_unreadable_config_is_a_named_error() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::write(root.join("tsconfig.json"), "{ this is not json").unwrap();
        let error = owning_config(&resolver(), &root.join("app.tsx")).unwrap_err();
        assert!(
            error.contains("tsconfig.json") && error.contains("app.tsx"),
            "the message must name the config and the file: {error}"
        );
    }
}
