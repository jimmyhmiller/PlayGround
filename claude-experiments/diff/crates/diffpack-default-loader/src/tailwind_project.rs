//! Tailwind project discovery and candidate-source scan policy.
//!
//! This layer owns filesystem/config behavior around the native Tailwind compiler;
//! the compiler itself remains in [crate::tailwind].

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use diffpack_core::CancelToken;

pub fn tailwind_scan_root(css_path: &Path, source_css: &str) -> PathBuf {
    let css_dir = css_path.parent().unwrap_or_else(|| Path::new("."));
    tailwind_source_root(source_css)
        .map(|rel| css_dir.join(rel))
        .unwrap_or_else(|| {
            let mut root = css_dir;
            for ancestor in css_dir.ancestors() {
                if ancestor.join("package.json").is_file() {
                    root = ancestor;
                    break;
                }
            }
            root.to_path_buf()
        })
}

/// The installed `tailwindcss` package directory Node resolution reaches from a
/// stylesheet: the nearest ancestor of the STYLESHEET holding a
/// `node_modules/tailwindcss/theme.css`.
///
/// Anchored on the stylesheet, not on the candidate scan root. Module resolution
/// is defined against the importing file; a `source(...)` scan root is a
/// source-tree concept with no relation to it, and joining `node_modules` onto
/// it only found the install when the two happened to coincide — TanStack
/// Start's `src/styles/app.css` with `source('../')` scans `src/`, which holds
/// no `node_modules`, so every such app silently compiled against the vendored
/// theme and shipped a stale `--font-sans`. Walking up from the file is also
/// what makes pnpm's nested layout and a monorepo root install resolve.
pub fn installed_tailwind_dir(css_path: &Path) -> Option<PathBuf> {
    css_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .ancestors()
        .map(|dir| dir.join("node_modules/tailwindcss"))
        .find(|package| package.join("theme.css").is_file())
}

/// The app's own installed Tailwind default theme, when present. Compiling
/// against it matches the exact Tailwind version the reference build used
/// (default tokens like `--font-sans` changed between v4 releases); without
/// it the vendored copy in `src/tailwind_theme.css` applies.
pub fn app_tailwind_theme(css_path: &Path) -> Option<String> {
    let package = installed_tailwind_dir(css_path)?;
    fs::read_to_string(package.join("theme.css")).ok()
}

/// States which engine produced one Tailwind entry's stylesheet, once per
/// (entry, message). A build that compiles the same entry for several passes
/// (client + react-server) says it once.
///
/// Silence here is not an option: whether a sheet came from diffpack's native
/// engine or from the app's own `tailwindcss` decides what is in it, and a reader
/// who has to infer that from a pixel diff has been failed.
pub fn report_tailwind_engine(css_path: &Path, message: &str) {
    static REPORTED: Mutex<Option<BTreeSet<String>>> = Mutex::new(None);
    let line = format!("[tailwind] {}: {message}", css_path.display());
    let mut reported = REPORTED.lock().unwrap();
    if reported
        .get_or_insert_with(BTreeSet::new)
        .insert(line.clone())
    {
        eprintln!("{line}");
    }
}

/// The `version` field of an installed package's `package.json`.
fn installed_package_version(package: &Path) -> Option<String> {
    let manifest = fs::read_to_string(package.join("package.json")).ok()?;
    let value: serde_json::Value = serde_json::from_str(&manifest).ok()?;
    value.get("version")?.as_str().map(str::to_string)
}

/// Warns, once per differing version, when the app's installed `tailwindcss` is
/// not the release the vendored data came from. The installed `theme.css` is
/// still used (its tokens are what the app's own build would emit), but the
/// preflight and the version banner remain the vendored ones — a mixture that
/// exists in no released Tailwind, so it is stated rather than left to be
/// discovered as a pixel diff.
pub fn warn_on_tailwind_version_drift(package: &Path) {
    static WARNED: Mutex<Option<BTreeSet<String>>> = Mutex::new(None);
    let Some(installed) = installed_package_version(package) else {
        return;
    };
    if installed == crate::tailwind::VERSION {
        return;
    }
    let mut warned = WARNED.lock().unwrap();
    if !warned
        .get_or_insert_with(BTreeSet::new)
        .insert(installed.clone())
    {
        return;
    }
    eprintln!(
        "warning: {} is tailwindcss v{installed}, but diffpack's vendored Tailwind data is \
         v{}. Its theme tokens are used as installed; the preflight and version banner \
         remain v{}. Re-vendor src/tailwind_theme.css / src/tailwind_preflight*.css if the \
         output diverges.",
        package.display(),
        crate::tailwind::VERSION,
        crate::tailwind::VERSION,
    );
}

/// The full app theme fed to the Tailwind compiler: the installed `tailwindcss`
/// default `theme.css`, EXTENDED with the `@theme`/`@keyframes` tokens derived from a
/// legacy JS config referenced by a `@config '<path>'` directive in `css` (if any).
/// A `@config`-defined token overrides the default (it is appended after it).
pub fn app_tailwind_theme_full(scan_root: &Path, css: &str, css_path: &Path) -> Option<String> {
    let base = app_tailwind_theme(css_path);
    let config = at_config_theme(scan_root, css, css_path);
    match (base, config) {
        (Some(base), Some(cfg)) => Some(format!("{base}\n{cfg}")),
        (base, None) => base,
        // A v3 config with no installed `tailwindcss/theme.css`: merge the config tokens
        // ON TOP of the vendored default theme so the config EXTENDS the default scale
        // rather than replacing it (a bare `--color-brand` must not drop `p-4`/`flex`).
        (None, Some(cfg)) => Some(format!(
            "{}\n{}",
            crate::tailwind::vendored_theme_css(),
            cfg
        )),
    }
}

/// The path string in a `@config '<path>'` / `@config "<path>"` directive, if present.
fn parse_at_config(css: &str) -> Option<String> {
    let after = &css[css.find("@config")? + "@config".len()..];
    let open = after.find(['\'', '"'])?;
    let quote = after.as_bytes()[open] as char;
    let inner = &after[open + 1..];
    let close = inner.find(quote)?;
    Some(inner[..close].to_string())
}

/// Evaluate a `@config`-referenced legacy JS Tailwind config (via node + the app's
/// own jiti) into v4 `@theme`/`@keyframes` CSS. Returns `None` when there is no
/// `@config`, the config file is missing, or node is unavailable — the compile then
/// proceeds on the default theme (a `@config` on a config with only content/plugins
/// contributes no theme tokens anyway). Never silently mis-maps: the node evaluator
/// reports unmapped theme categories on stderr, surfaced here.
/// Discovers a legacy v3 `tailwind.config.{js,cjs,mjs,ts}` at the project scan root
/// (v3 apps declare the config there, with no `@config` directive in the CSS). Returns
/// the first that exists.
fn discover_v3_config(scan_root: &Path) -> Option<PathBuf> {
    [
        "tailwind.config.js",
        "tailwind.config.cjs",
        "tailwind.config.mjs",
        "tailwind.config.ts",
    ]
    .iter()
    .map(|name| scan_root.join(name))
    .find(|p| p.exists())
}

fn at_config_theme(scan_root: &Path, css: &str, css_path: &Path) -> Option<String> {
    // A `@config '<path>'` directive names the config explicitly (v4-style). Otherwise a
    // legacy v3 entry auto-discovers `tailwind.config.*` at the scan root — but a v4
    // entry with no `@config` uses NO JS config (so a stray tailwind.config.js is not
    // picked up for it).
    let config_path = match parse_at_config(css) {
        Some(rel) => css_path.parent()?.join(rel),
        None if crate::tailwind::is_tailwind_v3_entry(css) => discover_v3_config(scan_root)?,
        None => return None,
    };
    if !config_path.exists() {
        eprintln!(
            "[tailwind @config] config file not found: {} (theme tokens from it will be missing)",
            config_path.display()
        );
        return None;
    }
    // The evaluator resolves jiti + the config's imports from the CONFIG's
    // node_modules, so it can live in a temp file; run it from the config's dir.
    let loader = std::env::temp_dir().join("diffpack-tailwind-config-eval.mjs");
    if fs::write(
        &loader,
        include_str!("../../../scripts/tailwind-config-eval.mjs"),
    )
    .is_err()
    {
        return None;
    }
    let output = std::process::Command::new("node")
        .arg(&loader)
        .arg(&config_path)
        .current_dir(config_path.parent().unwrap_or_else(|| Path::new(".")))
        .output()
        .ok()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.trim().is_empty() {
        eprintln!("[tailwind @config] {}", stderr.trim());
    }
    if !output.status.success() {
        return None;
    }
    let theme = String::from_utf8_lossy(&output.stdout).to_string();
    (!theme.trim().is_empty()).then_some(theme)
}

/// Parses the `source('...')` argument of a Tailwind v4 `@import 'tailwindcss'`
/// entry: the (entry-relative) directory the compiler scans for classes.
fn tailwind_source_root(source_css: &str) -> Option<String> {
    let start = source_css.find("source(")? + "source(".len();
    let rest = &source_css[start..];
    let end = rest.find(')')?;
    Some(rest[..end].trim().trim_matches(['\'', '"']).to_string())
}

/// What the candidate scan must not descend into: the scan root's `.gitignore`
/// entries (Tailwind's own scanner respects `.gitignore`, which is how a
/// checked-in reference build never picks candidates out of `dist/`) plus this
/// build's own output directory (never scan what we emitted).
pub struct ScanSkip {
    /// Simple ignored names (`dist`, `logs`): skipped wherever they appear.
    names: Vec<String>,
    /// Ignored filename suffixes from `*.<ext>`-style patterns (`.log`).
    suffixes: Vec<String>,
    /// The build's canonical output root.
    out_root: Option<PathBuf>,
    /// `@source not "<glob>"` exclusions, brace-expanded and split into
    /// segments. A path matching any of them is skipped wherever the scan
    /// reaches it.
    excluded: Vec<Vec<String>>,
}

impl ScanSkip {
    pub fn for_root(scan_root: &Path, out_root: &Path) -> ScanSkip {
        let mut names = Vec::new();
        let mut suffixes = Vec::new();
        if let Ok(gitignore) = fs::read_to_string(scan_root.join(".gitignore")) {
            for line in gitignore.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') || line.starts_with('!') {
                    continue;
                }
                let entry = line.trim_matches('/');
                if let Some(suffix) = entry.strip_prefix("*.") {
                    if !suffix.contains(['*', '/', '?', '[']) {
                        suffixes.push(format!(".{suffix}"));
                    }
                } else if !entry.contains(['*', '/', '?', '[']) {
                    names.push(entry.to_string());
                }
            }
        }
        ScanSkip {
            names,
            suffixes,
            out_root: fs::canonicalize(out_root).ok(),
            excluded: Vec::new(),
        }
    }

    /// Replaces the brace-expanded @source exclusions used by this scan.
    pub fn set_excluded(&mut self, patterns: &[String]) {
        self.excluded = patterns
            .iter()
            .flat_map(|pattern| expand_braces(pattern))
            .map(|pattern| path_segments(Path::new(&pattern)))
            .collect();
    }

    /// Whether the walk skips this entry. `is_dir` comes from the directory entry itself
    /// (never a fresh `stat`), and gates the out-root check below: the build's output root is
    /// a directory, so a FILE can never be it, and canonicalizing every file to find that out
    /// cost more than the rest of the walk put together on a monorepo.
    fn skips(&self, path: &Path, name: &str, is_dir: bool) -> bool {
        if name.starts_with('.') || name == "node_modules" {
            return true;
        }
        if self.names.iter().any(|n| n == name)
            || self.suffixes.iter().any(|s| name.ends_with(s.as_str()))
        {
            return true;
        }
        if !self.excluded.is_empty() {
            let segments = path_segments(path);
            if self
                .excluded
                .iter()
                .any(|pattern| glob_matches(pattern, &segments))
            {
                return true;
            }
        }
        if is_dir
            && let Some(out_root) = &self.out_root
            && let Ok(canonical) = fs::canonicalize(path)
            && canonical == *out_root
        {
            return true;
        }
        false
    }
}

/// The `@source` directives of a compiled Tailwind entry, split into the extra
/// paths to scan and the `not`-negated paths to exclude. Every path is absolute
/// (`css::absolutize_source_directives` anchors each one to the file that wrote
/// it before the entry's imports are spliced together).
pub fn tailwind_source_globs(css: &str) -> Result<(Vec<String>, Vec<String>), String> {
    let mut included = Vec::new();
    let mut excluded = Vec::new();
    let mut rest = css;
    while let Some(at) = rest.find("@source") {
        let body = &rest[at + "@source".len()..];
        let Some(end) = body.find(';') else {
            break;
        };
        let statement = body[..end].trim();
        rest = &body[end + 1..];
        let (negated, target) = match statement.strip_prefix("not") {
            Some(tail) if tail.starts_with(char::is_whitespace) => (true, tail.trim()),
            _ => (false, statement),
        };
        let Some(path) = target
            .strip_prefix(['"', '\''])
            .and_then(|value| value.get(..value.len().saturating_sub(1)))
        else {
            return Err(format!(
                "@source must name a quoted path (got `@source {statement}`)"
            ));
        };
        if path.contains('[') {
            return Err(format!(
                "`@source \"{path}\"` uses a character class, which diffpack's Tailwind \
                 source matcher does not implement (it supports `**`, `*`, `?` and `{{a,b}}`)"
            ));
        }
        if negated {
            excluded.push(path.to_string());
        } else {
            included.push(path.to_string());
        }
    }
    Ok((included, excluded))
}

/// Splits a path into its components as strings, for glob matching.
fn path_segments(path: &Path) -> Vec<String> {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect()
}

/// Expands `{a,b}` alternations into the concrete patterns they stand for.
/// Nested and repeated groups expand as the product, matching shell/glob
/// semantics.
fn expand_braces(pattern: &str) -> Vec<String> {
    let Some(open) = pattern.find('{') else {
        return vec![pattern.to_string()];
    };
    let mut depth = 0usize;
    let mut close = None;
    for (offset, byte) in pattern[open..].bytes().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(open + offset);
                    break;
                }
            }
            _ => {}
        }
    }
    let Some(close) = close else {
        return vec![pattern.to_string()];
    };
    let mut alternatives = Vec::new();
    let mut depth = 0usize;
    let mut start = open + 1;
    let inner = &pattern[open + 1..close];
    for (offset, byte) in inner.bytes().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => depth -= 1,
            b',' if depth == 0 => {
                alternatives.push(&pattern[start..open + 1 + offset]);
                start = open + 1 + offset + 1;
            }
            _ => {}
        }
    }
    alternatives.push(&pattern[start..close]);
    let mut out = Vec::new();
    for alternative in alternatives {
        let expanded = format!("{}{alternative}{}", &pattern[..open], &pattern[close + 1..]);
        out.extend(expand_braces(&expanded));
    }
    out
}

/// Whether a `*`/`?` pattern segment matches one path component. `*` matches any
/// run of characters within the component (never a `/`), `?` exactly one.
fn segment_matches(pattern: &str, name: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let name: Vec<char> = name.chars().collect();
    // Classic backtracking wildcard match, iterative so a pathological pattern
    // cannot blow the stack.
    let (mut p, mut n) = (0usize, 0usize);
    let (mut star, mut backtrack) = (None, 0usize);
    while n < name.len() {
        if p < pattern.len() && (pattern[p] == '?' || pattern[p] == name[n]) {
            p += 1;
            n += 1;
        } else if p < pattern.len() && pattern[p] == '*' {
            star = Some(p);
            backtrack = n;
            p += 1;
        } else if let Some(star) = star {
            p = star + 1;
            backtrack += 1;
            n = backtrack;
        } else {
            return false;
        }
    }
    while p < pattern.len() && pattern[p] == '*' {
        p += 1;
    }
    p == pattern.len()
}

/// Whether a brace-expanded, segment-split glob matches a path's segments.
/// `**` matches any number of segments (including none).
fn glob_matches(pattern: &[String], path: &[String]) -> bool {
    if pattern.is_empty() {
        return path.is_empty();
    }
    if pattern[0] == "**" {
        // `**` consumes zero or more segments; try each split point.
        for taken in 0..=path.len() {
            if glob_matches(&pattern[1..], &path[taken..]) {
                return true;
            }
        }
        return false;
    }
    if path.is_empty() {
        return false;
    }
    segment_matches(&pattern[0], &path[0]) && glob_matches(&pattern[1..], &path[1..])
}

/// Whether a path segment carries glob metacharacters.
fn is_glob_segment(segment: &str) -> bool {
    segment.contains(['*', '?', '{'])
}

/// Reads every source file an `@source` pattern selects. A pattern with no glob
/// metacharacters names a file (read directly) or a directory (walked whole,
/// exactly as Tailwind treats a bare `@source "./dir"`).
pub fn collect_glob_sources(pattern: &str, out: &mut Vec<PathBuf>, skip: &ScanSkip) {
    for expanded in expand_braces(pattern) {
        let segments = path_segments(Path::new(&expanded));
        let literal = segments.iter().take_while(|s| !is_glob_segment(s)).count();
        let root: PathBuf = segments[..literal].iter().collect();
        if literal == segments.len() {
            if root.is_dir() {
                // An `@source` directory walk is not cancellable: the caller checks
                // between patterns, which is granular enough for the handful an app
                // declares.
                collect_scan_sources(&root, out, skip, &CancelToken::never());
            } else if root.is_file() {
                out.push(root.clone());
            }
            continue;
        }
        collect_matching_sources(&root, &segments, out, skip);
    }
}

/// Walks `directory`, collecting every file whose full path matches `pattern`.
fn collect_matching_sources(
    directory: &Path,
    pattern: &[String],
    out: &mut Vec<PathBuf>,
    skip: &ScanSkip,
) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let is_dir = entry
            .file_type()
            .map(|kind| kind.is_dir())
            .unwrap_or_else(|_| path.is_dir());
        if skip.skips(&path, &name, is_dir) {
            continue;
        }
        if is_dir {
            collect_matching_sources(&path, pattern, out, skip);
        } else if glob_matches(pattern, &path_segments(&path)) {
            out.push(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_globs_expand_braces_and_match_double_star_segments() {
        assert_eq!(
            expand_braces("a/*.{js,ts,tsx}"),
            vec![
                "a/*.js".to_string(),
                "a/*.ts".to_string(),
                "a/*.tsx".to_string()
            ]
        );
        assert_eq!(expand_braces("{a,b}/{x,y}").len(), 4);
        let pattern = |value: &str| path_segments(Path::new(value));
        let matches = |glob: &str, path: &str| glob_matches(&pattern(glob), &pattern(path));
        assert!(matches("/a/**/*.tsx", "/a/b/c/d.tsx"));
        assert!(matches("/a/**/*.tsx", "/a/d.tsx"));
        assert!(!matches("/a/**/*.tsx", "/a/b/c/d.ts"));
        assert!(!matches("/a/*.tsx", "/a/b/c.tsx"));
        assert!(segment_matches("*.ts?", "main.tsx"));
        assert!(!segment_matches("*.ts?", "main.ts"));
    }
}

/// Recursively gathers the sources the utility-class candidate scan reads:
/// every JS/TS/JSX/HTML file under the scan root. Skips `node_modules`,
/// dot-directories, `.gitignore`d entries (as Tailwind does), and the build's
/// own output directory, so only the app's own classes are scanned. The files
/// are scanned together (`scan_class_candidates_multi`) so identifiers resolve
/// across module boundaries.
/// Collects the PATH of every scannable source under `root` into `out`. Returns false if
/// `cancel` fired part-way, in which case `out` is incomplete and must not be scanned.
///
/// Deliberately does not read the files: the walk is a serial directory traversal, but
/// reading is per-file independent work and there are thousands of them (cal.com: 611 ms of
/// the 900 ms candidate scan was `read_to_string`), so the caller reads them in parallel.
pub fn collect_scan_sources(
    root: &Path,
    out: &mut Vec<PathBuf>,
    skip: &ScanSkip,
    cancel: &CancelToken<'_>,
) -> bool {
    let Ok(entries) = fs::read_dir(root) else {
        return true;
    };
    for entry in entries.flatten() {
        if cancel.cancelled() {
            return false;
        }
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let is_dir = entry
            .file_type()
            .map(|kind| kind.is_dir())
            .unwrap_or_else(|_| path.is_dir());
        if skip.skips(&path, &name, is_dir) {
            continue;
        }
        if is_dir {
            if !collect_scan_sources(&path, out, skip, cancel) {
                return false;
            }
        } else if matches!(
            path.extension().and_then(|value| value.to_str()),
            Some("js" | "jsx" | "ts" | "tsx" | "mjs" | "cjs" | "html")
        ) {
            out.push(path);
        }
    }
    true
}
