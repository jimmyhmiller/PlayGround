//! `sideEffects`-aware droppability, the authority the generic dead-module
//! elimination pass ([`crate::bundler`]) consults to decide whether a
//! reachable-but-unused module may be dropped from a build.
//!
//! A module's droppability is read from the nearest ancestor `package.json`,
//! matching Rollup/esbuild/webpack semantics:
//!
//! - field **absent** -> the module **has side effects** (the conservative
//!   default; such a module is never dropped),
//! - `false` -> the module has **no** side effects (droppable when unused),
//! - `true` -> the module **has** side effects,
//! - **array of globs** (or a single string) -> only files matching a glob have
//!   side effects; every other file in the package is droppable when unused.
//!
//! Glob support is deliberately faithful: `*` (one path segment), `**` (any
//! number of segments), `?` (one non-`/` character), and literals are matched
//! exactly. A glob shape this cannot evaluate (brace expansion `{a,b}`,
//! character classes `[...]`, or extglobs `?(...)`) is a hard, specific error
//! naming the glob — never a silent wrong match.
//!
//! The nearest `package.json`'s parsed `sideEffects` field is cached per
//! directory (keyed by absolute path, immutable for a build), so classifying a
//! whole package's modules reads each `package.json` once.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// The parsed `sideEffects` field of one `package.json`.
#[derive(Debug, Clone)]
enum SideEffectsField {
    /// A `package.json` with no `sideEffects` field: the package's modules have
    /// side effects (the conservative default).
    Absent,
    /// `"sideEffects": true | false`.
    Flag(bool),
    /// `"sideEffects": "glob"` or `["glob", ...]`: a file has side effects iff it
    /// matches one of these globs.
    Globs(Vec<String>),
}

/// Determines whether the module at `path` may be dropped when unused, per the
/// nearest ancestor `package.json`'s `sideEffects` field.
///
/// Returns `Ok(true)` when the module is droppable (declared free of side
/// effects), `Ok(false)` when it must be kept (has, or may have, side effects).
/// An unsupported glob shape in the governing `package.json` is a hard error
/// naming the glob.
pub fn is_droppable(path: &Path) -> Result<bool, String> {
    let Some((package_dir, field)) = nearest_side_effects(path) else {
        // No governing `package.json` at all: keep the module (conservative).
        return Ok(false);
    };
    match field {
        SideEffectsField::Absent | SideEffectsField::Flag(true) => Ok(false),
        SideEffectsField::Flag(false) => Ok(true),
        SideEffectsField::Globs(globs) => {
            let relative = path.strip_prefix(&package_dir).unwrap_or(path);
            let relative = relative
                .components()
                .filter_map(|component| component.as_os_str().to_str())
                .collect::<Vec<_>>()
                .join("/");
            for glob in &globs {
                if glob_matches(glob, &relative)? {
                    // Matches a side-effectful glob: not droppable.
                    return Ok(false);
                }
            }
            Ok(true)
        }
    }
}

fn cache() -> &'static Mutex<HashMap<PathBuf, Option<SideEffectsField>>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, Option<SideEffectsField>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Walks up from `path`'s directory to the nearest `package.json` and returns
/// its directory plus parsed `sideEffects` field. `None` when no ancestor
/// `package.json` exists.
fn nearest_side_effects(path: &Path) -> Option<(PathBuf, SideEffectsField)> {
    let mut directory = path.parent();
    while let Some(dir) = directory {
        if let Some(field) = side_effects_at(dir) {
            return Some((dir.to_path_buf(), field));
        }
        directory = dir.parent();
    }
    None
}

/// The parsed `sideEffects` field of the `package.json` directly in `dir`, or
/// `None` if `dir` has no `package.json`. Cached per directory.
fn side_effects_at(dir: &Path) -> Option<SideEffectsField> {
    if let Some(cached) = cache().lock().unwrap().get(dir) {
        return cached.clone();
    }
    let manifest = dir.join("package.json");
    let parsed = if manifest.is_file() {
        Some(parse_side_effects(&manifest))
    } else {
        None
    };
    cache()
        .lock()
        .unwrap()
        .insert(dir.to_path_buf(), parsed.clone());
    parsed
}

/// Reads and parses the `sideEffects` field of one `package.json`. A missing or
/// unreadable/unparseable file, or a field of an unexpected JSON shape, is
/// treated as [`SideEffectsField::Absent`] (the conservative default) rather
/// than an error, so a malformed dependency manifest never fails the build — it
/// just keeps that package's modules.
fn parse_side_effects(manifest: &Path) -> SideEffectsField {
    let Ok(text) = std::fs::read_to_string(manifest) else {
        return SideEffectsField::Absent;
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return SideEffectsField::Absent;
    };
    match value.get("sideEffects") {
        None | Some(serde_json::Value::Null) => SideEffectsField::Absent,
        Some(serde_json::Value::Bool(flag)) => SideEffectsField::Flag(*flag),
        Some(serde_json::Value::String(glob)) => SideEffectsField::Globs(vec![glob.clone()]),
        Some(serde_json::Value::Array(entries)) => SideEffectsField::Globs(
            entries
                .iter()
                .filter_map(|entry| entry.as_str().map(str::to_string))
                .collect(),
        ),
        // A number/object `sideEffects` is not a shape any tool defines; keep the
        // package's modules rather than guess.
        Some(_) => SideEffectsField::Absent,
    }
}

/// Whether `glob` (a `package.json` `sideEffects` entry) matches the
/// package-relative `path` (`/`-separated, no leading `./`).
///
/// A glob with no `/` matches against the file's basename (webpack semantics:
/// `"*.css"` marks every CSS file, at any depth); otherwise it matches the full
/// relative path. Supported tokens are `**`, `*`, `?`, and literals. Any other
/// glob construct is a hard error naming the glob.
fn glob_matches(glob: &str, path: &str) -> Result<bool, String> {
    let glob = glob.strip_prefix("./").unwrap_or(glob);
    validate_glob(glob)?;
    let pattern = glob.chars().collect::<Vec<_>>();
    if glob.contains('/') {
        Ok(glob_match(&pattern, &path.chars().collect::<Vec<_>>()))
    } else {
        let basename = path.rsplit('/').next().unwrap_or(path);
        Ok(glob_match(&pattern, &basename.chars().collect::<Vec<_>>()))
    }
}

/// Rejects glob constructs this matcher cannot faithfully evaluate, so an
/// unsupported shape is a clear error rather than a silent mismatch.
fn validate_glob(glob: &str) -> Result<(), String> {
    // Brace expansion (`{a,b}`), character classes (`[...]`), extglobs (`?(...)`,
    // via the `(`), and negation (`!`) are not evaluated by this matcher — flag
    // any of them rather than silently mismatch. `+`/`@` outside a `(` are plain
    // literals, so only the grouping/negation characters are rejected.
    for character in glob.chars() {
        if matches!(character, '{' | '}' | '[' | ']' | '(' | ')' | '!') {
            return Err(format!(
                "unsupported `sideEffects` glob {glob:?}: the {character:?} construct \
                 (brace expansion, character class, extglob, or negation) is not supported"
            ));
        }
    }
    Ok(())
}

/// Full-match glob evaluation. `**` matches any run of characters (including
/// `/`); `*` matches a run of non-`/` characters; `?` matches one non-`/`
/// character; everything else is a literal.
fn glob_match(pattern: &[char], text: &[char]) -> bool {
    match pattern.first() {
        None => text.is_empty(),
        Some('*') => {
            if pattern.get(1) == Some(&'*') {
                let rest = &pattern[2..];
                // `**` matches any run of characters, including `/`.
                if (0..=text.len()).any(|consumed| glob_match(rest, &text[consumed..])) {
                    return true;
                }
                // `**/` additionally matches zero directories, so `**/*.scss`
                // matches a bare `x.scss` — the trailing slash may be elided.
                rest.first() == Some(&'/') && glob_match(&rest[1..], text)
            } else {
                let rest = &pattern[1..];
                let mut consumed = 0;
                loop {
                    if glob_match(rest, &text[consumed..]) {
                        return true;
                    }
                    if consumed >= text.len() || text[consumed] == '/' {
                        return false;
                    }
                    consumed += 1;
                }
            }
        }
        Some('?') => {
            !text.is_empty() && text[0] != '/' && glob_match(&pattern[1..], &text[1..])
        }
        Some(&literal) => {
            !text.is_empty() && text[0] == literal && glob_match(&pattern[1..], &text[1..])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_star_stays_within_a_segment() {
        assert!(glob_matches("*.css", "foo.css").unwrap());
        assert!(glob_matches("*.css", "a/b/foo.css").unwrap()); // no slash -> basename
        assert!(!glob_matches("*.css", "foo.scss").unwrap());
        assert!(glob_matches("src/*.js", "src/x.js").unwrap());
        assert!(!glob_matches("src/*.js", "src/nested/x.js").unwrap());
    }

    #[test]
    fn glob_double_star_crosses_segments() {
        assert!(glob_matches("**/*.scss", "a/b/c/x.scss").unwrap());
        assert!(glob_matches("**/*.scss", "x.scss").unwrap());
        assert!(glob_matches("src/**/*.js", "src/a/b/x.js").unwrap());
        assert!(!glob_matches("src/**/*.js", "lib/a/x.js").unwrap());
    }

    #[test]
    fn glob_exact_relative_path() {
        assert!(glob_matches("./src/x.js", "src/x.js").unwrap());
        assert!(!glob_matches("./src/x.js", "src/y.js").unwrap());
    }

    #[test]
    fn glob_question_matches_one_non_slash_character() {
        assert!(glob_matches("a?c.js", "abc.js").unwrap());
        assert!(!glob_matches("a?c.js", "ac.js").unwrap());
    }

    #[test]
    fn unsupported_glob_shapes_are_hard_errors() {
        assert!(glob_matches("*.{css,scss}", "a.css").is_err());
        assert!(glob_matches("*.[jt]s", "a.js").is_err());
        assert!(glob_matches("!(*.css)", "a.js").is_err());
    }

    #[test]
    fn classifies_from_nearest_package_json() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        // `false` -> droppable.
        std::fs::write(
            root.join("package.json"),
            r#"{ "name": "p", "sideEffects": false }"#,
        )
        .unwrap();
        std::fs::create_dir_all(root.join("dist")).unwrap();
        let module = root.join("dist/index.js");
        std::fs::write(&module, "export const x = 1;").unwrap();
        assert!(is_droppable(&module).unwrap());
    }

    #[test]
    fn absent_field_and_true_flag_keep_the_module() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("package.json"), r#"{ "name": "keep-absent" }"#).unwrap();
        let module = root.join("index.js");
        std::fs::write(&module, "x").unwrap();
        assert!(!is_droppable(&module).unwrap());

        let dir2 = tempfile::tempdir().unwrap();
        let root2 = dir2.path();
        std::fs::write(
            root2.join("package.json"),
            r#"{ "name": "keep-true", "sideEffects": true }"#,
        )
        .unwrap();
        let module2 = root2.join("index.js");
        std::fs::write(&module2, "x").unwrap();
        assert!(!is_droppable(&module2).unwrap());
    }

    #[test]
    fn glob_array_keeps_only_matching_files() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(
            root.join("package.json"),
            r#"{ "name": "g", "sideEffects": ["*.css", "./src/polyfill.js"] }"#,
        )
        .unwrap();
        std::fs::create_dir_all(root.join("src")).unwrap();

        let css = root.join("src/theme.css");
        std::fs::write(&css, "").unwrap();
        assert!(!is_droppable(&css).unwrap(), "a matched CSS file has side effects");

        let polyfill = root.join("src/polyfill.js");
        std::fs::write(&polyfill, "").unwrap();
        assert!(!is_droppable(&polyfill).unwrap(), "an exact-path match has side effects");

        let plain = root.join("src/util.js");
        std::fs::write(&plain, "").unwrap();
        assert!(is_droppable(&plain).unwrap(), "an unmatched file is droppable");
    }
}
