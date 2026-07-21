//! Native CSS processing for the bundler's stylesheet pipeline.
//!
//! Three loaders share this module:
//!
//! - **Global stylesheets** (`import "./app.css"`): top-level `@import`
//!   statements are extracted into real graph dependency edges (so edits to the
//!   imported file invalidate correctly and the emitted concatenation is
//!   imported-before-importer, deduped once per graph like CSS spec order), and
//!   every relative `url(...)` is resolved against the CSS file, content-hashed,
//!   and rewritten to its emitted public URL. Everything else passes through
//!   verbatim.
//! - **CSS Modules** (`*.module.css`): every local class/id selector and
//!   `@keyframes` name is rewritten to a deterministic scoped name
//!   `_<local>_<hash>` (hash of the file name + content, stable across rebuilds
//!   of unchanged content), `:global(...)`/`:local(...)` wrappers are honoured
//!   and removed, `composes:` declarations are folded into the exported mapping
//!   (cross-file composes become import edges resolved at runtime, so editing
//!   the composed file needs no re-derivation of the composer), and anything the
//!   scoper cannot handle confidently is a hard error naming the file and the
//!   construct — never a silent pass-through of unscoped locals.
//! - **Media-qualified imports** (`@import './x.css' screen;`): the target
//!   becomes a distinct `x.css?media=screen` graph module whose emitted CSS is
//!   the target's content (with its own imports/urls processed relative to
//!   itself, nested imports inlined) wrapped in `@media screen { ... }`. The
//!   inlined nested files are reported so the bundler can re-derive the module
//!   when any of them changes.
//!
//! The parser is a small hand-rolled tokenizer (strings/comments/paren aware),
//! deliberately owned code rather than a CSS crate dependency.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::bundler::{asset_public_name, content_hash};

/// One `@import` extracted from a CSS file, destined to become a real graph
/// dependency edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CssImport {
    /// The import target as a resolver specifier: `./x.css` for a relative (or
    /// relative-existing bare) target, or a bare package specifier
    /// (`some-package/theme.css`) resolved through the module resolver.
    pub specifier: String,
    /// The media query suffix (`@import './x.css' screen;`), when present. The
    /// bundler encodes it into the dependency's module id (`./x.css?media=screen`)
    /// so the imported content is emitted wrapped in `@media screen { ... }`.
    pub media: Option<String>,
}

/// One `url(...)` asset reference resolved from a CSS file: the absolute source
/// path and the content-hashed public name the reference was rewritten to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CssAsset {
    pub source: PathBuf,
    pub public_name: String,
}

/// One segment of a CSS Module mapping value. A local's exported value is the
/// space-joined concatenation of its segments: its own scoped name first, then
/// everything it composes (in declaration order, same-file chains resolved
/// transitively).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MappingSegment {
    /// A literal scoped (or `composes: x from global` unscoped) class name.
    Literal(String),
    /// A name looked up at runtime from another CSS module's mapping: index into
    /// [`ProcessedCss::compose_imports`] plus the original local name there.
    Foreign { import: usize, name: String },
}

/// The export mapping of a CSS Module: ordered `local name -> mapping value`
/// entries (deterministic: sorted by name).
pub type CssMapping = Vec<(String, Vec<MappingSegment>)>;

/// The result of processing one CSS file.
#[derive(Debug, Clone, Default)]
pub struct ProcessedCss {
    /// The final stylesheet text: imports stripped, locals scoped (modules),
    /// `url(...)` references rewritten to public asset URLs.
    pub css: String,
    /// `@import` targets that must become graph dependency edges.
    pub imports: Vec<CssImport>,
    /// Remote/absolute `@import` statements (scheme or `//` URLs) reproduced
    /// verbatim; the emit step hoists them, deduped, to the very top of the
    /// emitted stylesheet (an `@import` is only valid before all rules).
    pub external_imports: Vec<String>,
    /// Every asset referenced by a rewritten `url(...)`.
    pub assets: Vec<CssAsset>,
    /// Physical files (other than the processed file itself) whose content was
    /// inlined; edits to any of them must re-derive the owning module.
    pub inlined_files: Vec<PathBuf>,
    /// CSS Modules only: ordered `local name -> mapping value` entries
    /// (deterministic: sorted by name).
    pub mapping: CssMapping,
    /// CSS Modules only: the distinct `composes: x from './other.module.css'`
    /// source specifiers, in first-use order; `MappingSegment::Foreign.import`
    /// indexes into this.
    pub compose_imports: Vec<String>,
}

/// Whether a resolved path is a CSS Module (`*.module.css`).
pub fn is_css_module_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".module.css") && name != ".module.css")
}

/// Processes a plain (global) stylesheet: extracts top-level `@import`s and
/// rewrites `url(...)` references. No scoping; the rest of the text passes
/// through verbatim.
pub fn process_global_css(file: &Path, text: &str) -> Result<ProcessedCss, String> {
    let (css, imports, external_imports) = extract_top_level_imports(file, text)?;
    let mut assets = Vec::new();
    let css = rewrite_urls(&css, file, &mut assets)?;
    Ok(ProcessedCss {
        css,
        imports,
        external_imports,
        assets,
        ..ProcessedCss::default()
    })
}

/// Processes a CSS Module: scopes locals, resolves `composes`, extracts
/// `@import`s, rewrites `url(...)` references, and builds the export mapping.
pub fn process_css_module(file: &Path, text: &str) -> Result<ProcessedCss, String> {
    let display = file.display().to_string();
    let hash = scope_hash(file, text);
    let mut keyframes_names = BTreeSet::new();
    prescan_keyframes(text, &mut keyframes_names);
    let mut parser = ModuleParser {
        text,
        bytes: text.as_bytes(),
        display: &display,
        hash: &hash,
        pos: 0,
        out: Vec::with_capacity(text.len()),
        keyframes_names: &keyframes_names,
        class_locals: BTreeMap::new(),
        imports: Vec::new(),
        external_imports: Vec::new(),
        file,
    };
    parser.parse_items(true)?;
    let (mapping, compose_imports) = parser.resolve_mapping()?;
    let scoped = String::from_utf8(parser.out)
        .map_err(|error| format!("scoped CSS for {display} is not valid UTF-8: {error}"))?;
    let imports = parser.imports;
    let external_imports = parser.external_imports;
    let mut assets = Vec::new();
    let css = rewrite_urls(&scoped, file, &mut assets)?;
    Ok(ProcessedCss {
        css,
        imports,
        external_imports,
        assets,
        inlined_files: Vec::new(),
        mapping,
        compose_imports,
    })
}

/// Processes the target of a media-qualified `@import`: the file's content
/// (imports inlined recursively, urls resolved relative to each contributing
/// file, CSS Modules scoped) wrapped in `@media <media> { ... }`.
pub fn process_media_import(file: &Path, text: &str, media: &str) -> Result<ProcessedCss, String> {
    let mut assets = Vec::new();
    let mut inlined_files = Vec::new();
    let mut visited = BTreeSet::new();
    visited.insert(canonical_path(file));
    let inner = inline_file(file, text, &mut visited, &mut assets, &mut inlined_files)?;
    let css = format!("@media {media} {{\n{inner}}}\n");
    Ok(ProcessedCss {
        css,
        assets,
        inlined_files,
        ..ProcessedCss::default()
    })
}

/// Recursively inlines a CSS file and its `@import` graph for a media-qualified
/// import module. Each file's `url(...)`s resolve relative to THAT file; nested
/// media-qualified imports become nested `@media` blocks; a file is inlined at
/// most once (CSS-spec-like dedup); anything that cannot be inlined here (a
/// bare package or remote import) is a hard error naming it.
fn inline_file(
    file: &Path,
    text: &str,
    visited: &mut BTreeSet<PathBuf>,
    assets: &mut Vec<CssAsset>,
    inlined_files: &mut Vec<PathBuf>,
) -> Result<String, String> {
    let processed = if is_css_module_path(file) {
        process_css_module(file, text)?
    } else {
        process_global_css(file, text)?
    };
    if let Some(external) = processed.external_imports.first() {
        return Err(format!(
            "remote @import {external:?} inside a media-qualified @import is not supported \
             (in {})",
            file.display()
        ));
    }
    let mut out = String::new();
    let directory = file.parent().unwrap_or_else(|| Path::new("."));
    for import in &processed.imports {
        let relative = import
            .specifier
            .strip_prefix("./")
            .map(str::to_owned)
            .or_else(|| {
                import
                    .specifier
                    .starts_with("../")
                    .then(|| import.specifier.clone())
            });
        let Some(relative) = relative else {
            return Err(format!(
                "@import {:?} inside a media-qualified @import must be a relative path \
                 (in {})",
                import.specifier,
                file.display()
            ));
        };
        let target = canonical_path(&directory.join(relative));
        if !visited.insert(target.clone()) {
            continue;
        }
        let nested_text = fs::read_to_string(&target).map_err(|error| {
            format!(
                "cannot read @import {:?} (from {}): {}: {error}",
                import.specifier,
                file.display(),
                target.display()
            )
        })?;
        inlined_files.push(target.clone());
        let nested = inline_file(&target, &nested_text, visited, assets, inlined_files)?;
        match &import.media {
            Some(media) => {
                out.push_str(&format!("@media {media} {{\n{nested}}}\n"));
            }
            None => out.push_str(&nested),
        }
    }
    assets.extend(processed.assets);
    out.push_str(&processed.css);
    if !out.ends_with('\n') {
        out.push('\n');
    }
    Ok(out)
}

fn canonical_path(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

/// The deterministic scope suffix for one CSS Module: derived from the file's
/// name and content, so it is stable across rebuilds of unchanged content and
/// changes when the content does.
fn scope_hash(file: &Path, text: &str) -> String {
    let name = file
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("module.css");
    let mut identity = String::with_capacity(name.len() + 1 + text.len());
    identity.push_str(name);
    identity.push('\0');
    identity.push_str(text);
    format!("{:08x}", content_hash(identity.as_bytes()) & 0xffff_ffff)
}

// ---------------------------------------------------------------------------
// Low-level tokenizer helpers (strings/comments/paren aware).
// ---------------------------------------------------------------------------

pub(crate) fn is_ident_byte(byte: u8) -> bool {
    byte == b'-' || byte == b'_' || byte.is_ascii_alphanumeric() || byte >= 0x80
}

/// End index of the identifier starting at `start` (may equal `start` when no
/// identifier is present).
pub(crate) fn ident_end(bytes: &[u8], start: usize) -> usize {
    let mut index = start;
    while index < bytes.len() && is_ident_byte(bytes[index]) {
        index += 1;
    }
    index
}

pub(crate) fn skip_ws(bytes: &[u8], mut index: usize) -> usize {
    while index < bytes.len() && bytes[index].is_ascii_whitespace() {
        index += 1;
    }
    index
}

/// Index past the closing quote of the string starting at `index` (which must
/// hold a quote). Backslash escapes are honoured; an unterminated string ends at
/// EOF.
pub(crate) fn skip_string(bytes: &[u8], index: usize) -> usize {
    let quote = bytes[index];
    let mut cursor = index + 1;
    while cursor < bytes.len() {
        match bytes[cursor] {
            b'\\' => cursor += 2,
            byte if byte == quote => return cursor + 1,
            _ => cursor += 1,
        }
    }
    bytes.len()
}

/// Index past the `*/` of the comment starting at `index` (which must hold
/// `/*`). An unterminated comment ends at EOF.
pub(crate) fn skip_comment(bytes: &[u8], index: usize) -> usize {
    let mut cursor = index + 2;
    while cursor + 1 < bytes.len() {
        if bytes[cursor] == b'*' && bytes[cursor + 1] == b'/' {
            return cursor + 2;
        }
        cursor += 1;
    }
    bytes.len()
}

pub(crate) fn at_comment(bytes: &[u8], index: usize) -> bool {
    bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'*')
}

pub(crate) fn skip_ws_and_comments(bytes: &[u8], mut index: usize) -> usize {
    loop {
        let advanced = skip_ws(bytes, index);
        if advanced < bytes.len() && at_comment(bytes, advanced) {
            index = skip_comment(bytes, advanced);
        } else {
            return advanced;
        }
    }
}

/// Index of the `close` byte matching the `open` byte at `index`, skipping
/// strings and comments and honouring nesting.
pub(crate) fn find_matching(
    bytes: &[u8],
    index: usize,
    open: u8,
    close: u8,
    file: &str,
) -> Result<usize, String> {
    let mut depth = 0usize;
    let mut cursor = index;
    while cursor < bytes.len() {
        match bytes[cursor] {
            b'"' | b'\'' => cursor = skip_string(bytes, cursor),
            b'/' if at_comment(bytes, cursor) => cursor = skip_comment(bytes, cursor),
            byte if byte == open => {
                depth += 1;
                cursor += 1;
            }
            byte if byte == close => {
                depth -= 1;
                if depth == 0 {
                    return Ok(cursor);
                }
                cursor += 1;
            }
            _ => cursor += 1,
        }
    }
    Err(format!(
        "unterminated {:?} block in {file}",
        char::from(open)
    ))
}

pub(crate) fn starts_with_ci(bytes: &[u8], index: usize, prefix: &str) -> bool {
    let prefix = prefix.as_bytes();
    bytes.len() >= index + prefix.len()
        && bytes[index..index + prefix.len()]
            .iter()
            .zip(prefix)
            .all(|(byte, expected)| byte.eq_ignore_ascii_case(expected))
}

// ---------------------------------------------------------------------------
// @import extraction (shared by global stylesheets and CSS Modules).
// ---------------------------------------------------------------------------

enum ParsedImport {
    Graph(CssImport),
    External(String),
}

/// Extracts every top-level `@import` from `text` (removing the statements from
/// the returned CSS) and classifies each as a graph edge or an external
/// (remote) import to hoist. Everything else is copied verbatim.
fn extract_top_level_imports(
    file: &Path,
    text: &str,
) -> Result<(String, Vec<CssImport>, Vec<String>), String> {
    let bytes = text.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut imports = Vec::new();
    let mut externals = Vec::new();
    let mut depth = 0usize;
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'"' | b'\'' => {
                let end = skip_string(bytes, index);
                out.extend_from_slice(&bytes[index..end]);
                index = end;
            }
            b'/' if at_comment(bytes, index) => {
                let end = skip_comment(bytes, index);
                out.extend_from_slice(&bytes[index..end]);
                index = end;
            }
            b'{' => {
                depth += 1;
                out.push(b'{');
                index += 1;
            }
            b'}' => {
                depth = depth.saturating_sub(1);
                out.push(b'}');
                index += 1;
            }
            b'@' if depth == 0
                && starts_with_ci(bytes, index + 1, "import")
                && bytes
                    .get(index + 1 + "import".len())
                    .is_none_or(|byte| !is_ident_byte(*byte)) =>
            {
                let statement_end = find_statement_end(bytes, index);
                let statement = text[index..statement_end].trim_end_matches(';').trim();
                let body = statement["@import".len()..].trim();
                match parse_import_statement(file, statement, body)? {
                    ParsedImport::Graph(import) => imports.push(import),
                    ParsedImport::External(external) => externals.push(external),
                }
                index = statement_end;
                // Swallow one trailing newline so removed statements do not
                // leave a stack of blank lines behind.
                if bytes.get(index) == Some(&b'\n') {
                    index += 1;
                }
            }
            _ => {
                out.push(bytes[index]);
                index += 1;
            }
        }
    }
    let css = String::from_utf8(out).map_err(|error| {
        format!(
            "CSS for {} is not valid UTF-8 after @import extraction: {error}",
            file.display()
        )
    })?;
    Ok((css, imports, externals))
}

/// Index past the `;` ending the statement starting at `index` (or EOF),
/// skipping strings, comments, and parenthesized groups.
fn find_statement_end(bytes: &[u8], index: usize) -> usize {
    let mut cursor = index;
    while cursor < bytes.len() {
        match bytes[cursor] {
            b'"' | b'\'' => cursor = skip_string(bytes, cursor),
            b'/' if at_comment(bytes, cursor) => cursor = skip_comment(bytes, cursor),
            b'(' => {
                cursor = match find_matching(bytes, cursor, b'(', b')', "@import") {
                    Ok(close) => close + 1,
                    Err(_) => bytes.len(),
                }
            }
            b';' => return cursor + 1,
            _ => cursor += 1,
        }
    }
    bytes.len()
}

/// Parses one `@import` statement body (everything after `@import`, without the
/// trailing `;`): the target (quoted string or `url(...)`) plus an optional
/// media query. Unsupported forms (`layer(...)`, `supports(...)`) are hard
/// errors naming the statement.
fn parse_import_statement(
    file: &Path,
    statement: &str,
    body: &str,
) -> Result<ParsedImport, String> {
    let bytes = body.as_bytes();
    let display = file.display();
    if body.is_empty() {
        return Err(format!("@import in {display} has no target ({statement:?})"));
    }
    let (target, rest) = if bytes[0] == b'"' || bytes[0] == b'\'' {
        let end = skip_string(bytes, 0);
        if end > body.len() || end < 2 {
            return Err(format!(
                "unterminated @import target in {display} ({statement:?})"
            ));
        }
        (body[1..end - 1].to_string(), body[end..].trim())
    } else if starts_with_ci(bytes, 0, "url(") {
        let close = find_matching(bytes, 3, b'(', b')', &display.to_string())?;
        let inner = body[4..close].trim();
        let inner = inner
            .strip_prefix(['"', '\''])
            .and_then(|value| value.strip_suffix(['"', '\'']))
            .unwrap_or(inner);
        (inner.to_string(), body[close + 1..].trim())
    } else {
        return Err(format!(
            "unsupported @import target in {display} ({statement:?}); expected a quoted \
             path or url(...)"
        ));
    };
    if target.is_empty() {
        return Err(format!(
            "@import in {display} has an empty target ({statement:?})"
        ));
    }
    // Remote / absolute-scheme imports cannot be inlined; they are reproduced
    // verbatim and hoisted to the top of the emitted stylesheet.
    if target.starts_with("//") || has_url_scheme(&target) {
        return Ok(ParsedImport::External(format!("{statement};")));
    }
    if target.starts_with('/') {
        return Err(format!(
            "root-relative @import {target:?} in {display} is not supported \
             ({statement:?}); use a relative path or a package specifier"
        ));
    }
    // Unsupported @import conditions are hard errors, not silent drops.
    let rest_lower = rest.to_ascii_lowercase();
    if rest_lower == "layer" || rest_lower.starts_with("layer(") || rest_lower.starts_with("layer ")
    {
        return Err(format!(
            "@import with a layer(...) condition is not supported in {display} \
             ({statement:?})"
        ));
    }
    if rest_lower.starts_with("supports(") || rest_lower.starts_with("supports ") {
        return Err(format!(
            "@import with a supports(...) condition is not supported in {display} \
             ({statement:?})"
        ));
    }
    let media = if rest.is_empty() {
        None
    } else {
        if rest.contains(['&', '?', '#', ';']) {
            return Err(format!(
                "cannot parse the media query {rest:?} of @import in {display} \
                 ({statement:?})"
            ));
        }
        Some(rest.to_string())
    };
    // CSS resolves a bare target relative to the importing file; the module
    // resolver treats it as a package. Prefer the CSS-native relative file when
    // it exists, otherwise pass the bare specifier to the package resolver.
    let specifier = if target.starts_with("./") || target.starts_with("../") {
        target
    } else {
        let directory = file.parent().unwrap_or_else(|| Path::new("."));
        if directory.join(&target).is_file() {
            format!("./{target}")
        } else {
            target
        }
    };
    Ok(ParsedImport::Graph(CssImport { specifier, media }))
}

/// Whether a url/import target is an absolute URL with a scheme (`http:`,
/// `https:`, `data:`, ...).
fn has_url_scheme(target: &str) -> bool {
    let Some(colon) = target.find(':') else {
        return false;
    };
    colon > 0
        && target.as_bytes()[..colon]
            .iter()
            .enumerate()
            .all(|(index, byte)| {
                byte.is_ascii_alphabetic()
                    || (index > 0 && (byte.is_ascii_digit() || matches!(byte, b'+' | b'.' | b'-')))
            })
}

// ---------------------------------------------------------------------------
// url(...) rewriting (shared by every CSS loader).
// ---------------------------------------------------------------------------

/// Rewrites every relative `url(...)` in `css` to its emitted public asset URL
/// (`/assets/<stem>-<hash>.<ext>`), resolving relative to `file`'s directory
/// and recording each referenced asset. Absolute URLs (scheme or `//`),
/// `data:`, `#fragment`-only, and `/`-rooted public references are left
/// untouched. A reference that does not resolve to a real file is a hard error
/// naming the CSS file and the reference.
fn rewrite_urls(css: &str, file: &Path, assets: &mut Vec<CssAsset>) -> Result<String, String> {
    let bytes = css.as_bytes();
    let directory = file.parent().unwrap_or_else(|| Path::new("."));
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'"' | b'\'' => {
                let end = skip_string(bytes, index);
                out.extend_from_slice(&bytes[index..end]);
                index = end;
            }
            b'/' if at_comment(bytes, index) => {
                let end = skip_comment(bytes, index);
                out.extend_from_slice(&bytes[index..end]);
                index = end;
            }
            _ if starts_with_ci(bytes, index, "url(")
                && (index == 0 || !is_ident_byte(bytes[index - 1])) =>
            {
                let close = find_matching(bytes, index + 3, b'(', b')', &file.display().to_string())?;
                let raw = css[index + 4..close].trim();
                let target = raw
                    .strip_prefix(['"', '\''])
                    .and_then(|value| value.strip_suffix(['"', '\'']))
                    .unwrap_or(raw);
                if target.is_empty()
                    || target.starts_with('#')
                    || target.starts_with('/')
                    || has_url_scheme(target)
                {
                    // Left as written: fragment-only, public-rooted, or
                    // absolute (including data: and //) URLs.
                    out.extend_from_slice(&bytes[index..close + 1]);
                } else {
                    // A `?query` / `#fragment` suffix survives onto the
                    // rewritten URL (fonts commonly carry `#iefix`).
                    let split = target.find(['?', '#']).unwrap_or(target.len());
                    let (path_part, suffix) = target.split_at(split);
                    let source = directory.join(path_part);
                    let contents = fs::read(&source).map_err(|error| {
                        format!(
                            "cannot resolve url({raw}) referenced from {}: {}: {error}",
                            file.display(),
                            source.display()
                        )
                    })?;
                    let public_name = asset_public_name(&source, content_hash(&contents));
                    // Relative to the emitted stylesheet (which sits beside the `assets/`
                    // directory), so the reference is correct under ANY public base —
                    // absolute `/assets/...` would break a site served from a subpath.
                    let rewritten = format!("url(\"assets/{public_name}{suffix}\")");
                    out.extend_from_slice(rewritten.as_bytes());
                    let source = canonical_path(&source);
                    if !assets
                        .iter()
                        .any(|existing| existing.public_name == public_name)
                    {
                        assets.push(CssAsset {
                            source,
                            public_name,
                        });
                    }
                }
                index = close + 1;
            }
            byte => {
                out.push(byte);
                index += 1;
            }
        }
    }
    String::from_utf8(out).map_err(|error| {
        format!(
            "CSS for {} is not valid UTF-8 after url rewriting: {error}",
            file.display()
        )
    })
}

// ---------------------------------------------------------------------------
// CSS Modules scoping.
// ---------------------------------------------------------------------------

/// One `composes:` reference recorded on a local class.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ComposeRef {
    /// `composes: other;` — another class in the same file.
    Local(String),
    /// `composes: other from global;` — an unscoped global class name.
    Global(String),
    /// `composes: other from './other.module.css';` — resolved at runtime from
    /// the other module's mapping.
    Foreign { specifier: String, name: String },
}

/// Collects `@keyframes` names ahead of the structural parse, so
/// `animation`/`animation-name` references can be rewritten even when they
/// precede the declaration. `:global(...)`-named keyframes are excluded.
fn prescan_keyframes(text: &str, names: &mut BTreeSet<String>) {
    let bytes = text.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'"' | b'\'' => index = skip_string(bytes, index),
            b'/' if at_comment(bytes, index) => index = skip_comment(bytes, index),
            b'@' => {
                let end = ident_end(bytes, index + 1);
                let name = text[index + 1..end].to_ascii_lowercase();
                index = end;
                if name == "keyframes" || name.ends_with("-keyframes") {
                    let cursor = skip_ws_and_comments(bytes, index);
                    if starts_with_ci(bytes, cursor, ":global(") {
                        continue;
                    }
                    let name_end = ident_end(bytes, cursor);
                    if name_end > cursor {
                        names.insert(text[cursor..name_end].to_string());
                        index = name_end;
                    }
                }
            }
            _ => index += 1,
        }
    }
}

struct ModuleParser<'a> {
    text: &'a str,
    bytes: &'a [u8],
    display: &'a str,
    hash: &'a str,
    pos: usize,
    out: Vec<u8>,
    /// Every locally-declared `@keyframes` name (from the prescan); the rewrite
    /// target set for `animation`/`animation-name` values.
    keyframes_names: &'a BTreeSet<String>,
    /// Every scoped class/id local, with its recorded `composes:` references.
    class_locals: BTreeMap<String, Vec<ComposeRef>>,
    imports: Vec<CssImport>,
    external_imports: Vec<String>,
    file: &'a Path,
}

impl ModuleParser<'_> {
    fn scoped(&self, name: &str) -> String {
        format!("_{name}_{}", self.hash)
    }

    /// Copies whitespace and comments verbatim.
    fn copy_trivia(&mut self) {
        loop {
            let advanced = skip_ws(self.bytes, self.pos);
            self.out
                .extend_from_slice(&self.bytes[self.pos..advanced]);
            self.pos = advanced;
            if self.pos < self.bytes.len() && at_comment(self.bytes, self.pos) {
                let end = skip_comment(self.bytes, self.pos);
                self.out.extend_from_slice(&self.bytes[self.pos..end]);
                self.pos = end;
            } else {
                return;
            }
        }
    }

    /// Parses a run of items (at-rules and style rules) until EOF (`top`) or
    /// the `}` closing the enclosing block (left unconsumed for the caller).
    fn parse_items(&mut self, top: bool) -> Result<(), String> {
        loop {
            self.copy_trivia();
            if self.pos >= self.bytes.len() {
                if top {
                    return Ok(());
                }
                return Err(format!("unterminated block in CSS module {}", self.display));
            }
            match self.bytes[self.pos] {
                b'}' => {
                    if top {
                        return Err(format!(
                            "unmatched `}}` in CSS module {}",
                            self.display
                        ));
                    }
                    return Ok(());
                }
                b'@' => self.parse_at_rule(top)?,
                _ => self.parse_style_rule()?,
            }
        }
    }

    fn parse_at_rule(&mut self, top: bool) -> Result<(), String> {
        let at = self.pos;
        let end = ident_end(self.bytes, at + 1);
        let raw_name = &self.text[at + 1..end];
        let name = raw_name.to_ascii_lowercase();
        match name.as_str() {
            "import" => {
                if !top {
                    return Err(format!(
                        "@import inside a block is not supported in CSS module {}",
                        self.display
                    ));
                }
                let statement_end = find_statement_end(self.bytes, at);
                let statement = self.text[at..statement_end].trim_end_matches(';').trim();
                let body = statement["@import".len()..].trim();
                match parse_import_statement(self.file, statement, body)? {
                    ParsedImport::Graph(import) => self.imports.push(import),
                    ParsedImport::External(external) => self.external_imports.push(external),
                }
                self.pos = statement_end;
                Ok(())
            }
            "charset" => {
                // Files are read and emitted as UTF-8; a utf-8 @charset is
                // redundant (and invalid anywhere but byte 0 of a file, which
                // concatenation cannot preserve), so it is dropped. Any other
                // charset cannot be honoured and is a hard error.
                let statement_end = find_statement_end(self.bytes, at);
                let statement = &self.text[at..statement_end];
                if statement.to_ascii_lowercase().contains("utf-8") {
                    self.pos = statement_end;
                    Ok(())
                } else {
                    Err(format!(
                        "unsupported @charset {statement:?} in CSS module {} (only utf-8)",
                        self.display
                    ))
                }
            }
            "media" | "supports" | "container" => self.parse_conditional_group(at),
            "layer" => {
                // `@layer a, b;` (statement) passes through; `@layer x { ... }`
                // recurses so the rules inside are scoped.
                let mut cursor = at;
                loop {
                    if cursor >= self.bytes.len() {
                        return Err(format!(
                            "unterminated @layer in CSS module {}",
                            self.display
                        ));
                    }
                    match self.bytes[cursor] {
                        b'{' => break,
                        b';' => {
                            self.out
                                .extend_from_slice(&self.bytes[at..cursor + 1]);
                            self.pos = cursor + 1;
                            return Ok(());
                        }
                        b'"' | b'\'' => cursor = skip_string(self.bytes, cursor),
                        b'/' if at_comment(self.bytes, cursor) => {
                            cursor = skip_comment(self.bytes, cursor);
                        }
                        _ => cursor += 1,
                    }
                }
                self.parse_conditional_group(at)
            }
            "font-face" | "page" | "property" | "counter-style" | "font-feature-values" => {
                // No class selectors can occur inside these; copy verbatim.
                let mut cursor = at;
                while cursor < self.bytes.len() && self.bytes[cursor] != b'{' {
                    cursor = match self.bytes[cursor] {
                        b'"' | b'\'' => skip_string(self.bytes, cursor),
                        b'/' if at_comment(self.bytes, cursor) => {
                            skip_comment(self.bytes, cursor)
                        }
                        _ => cursor + 1,
                    };
                }
                if cursor >= self.bytes.len() {
                    return Err(format!(
                        "unterminated @{name} in CSS module {}",
                        self.display
                    ));
                }
                let close = find_matching(self.bytes, cursor, b'{', b'}', self.display)?;
                self.out.extend_from_slice(&self.bytes[at..close + 1]);
                self.pos = close + 1;
                Ok(())
            }
            _ if name == "keyframes" || name.ends_with("-keyframes") => {
                self.parse_keyframes(at, end)
            }
            _ => Err(format!(
                "unsupported at-rule `@{raw_name}` in CSS module {}; cannot scope its \
                 contents confidently",
                self.display
            )),
        }
    }

    /// A conditional group rule (`@media`, `@supports`, `@container`, block
    /// `@layer`): prelude copied verbatim, contents recursed so nested style
    /// rules are scoped.
    fn parse_conditional_group(&mut self, at: usize) -> Result<(), String> {
        let mut cursor = at;
        while cursor < self.bytes.len() && self.bytes[cursor] != b'{' {
            cursor = match self.bytes[cursor] {
                b'"' | b'\'' => skip_string(self.bytes, cursor),
                b'/' if at_comment(self.bytes, cursor) => skip_comment(self.bytes, cursor),
                b';' => {
                    return Err(format!(
                        "unexpected `;` in at-rule prelude in CSS module {}",
                        self.display
                    ));
                }
                _ => cursor + 1,
            };
        }
        if cursor >= self.bytes.len() {
            return Err(format!(
                "unterminated at-rule in CSS module {}",
                self.display
            ));
        }
        self.out.extend_from_slice(&self.bytes[at..cursor + 1]);
        self.pos = cursor + 1;
        self.parse_items(false)?;
        // parse_items stops AT the closing brace.
        self.out.push(b'}');
        self.pos += 1;
        Ok(())
    }

    /// `@keyframes name { ... }`: the name is scoped (unless wrapped in
    /// `:global(...)`), the body is copied verbatim (frame selectors are
    /// percentages/from/to, not classes).
    fn parse_keyframes(&mut self, at: usize, name_start: usize) -> Result<(), String> {
        self.out.extend_from_slice(&self.bytes[at..name_start]);
        let mut cursor = skip_ws_and_comments(self.bytes, name_start);
        self.out.push(b' ');
        if starts_with_ci(self.bytes, cursor, ":global(") {
            cursor += ":global(".len();
            cursor = skip_ws(self.bytes, cursor);
            let end = ident_end(self.bytes, cursor);
            if end == cursor {
                return Err(format!(
                    "@keyframes :global(...) without a name in CSS module {}",
                    self.display
                ));
            }
            self.out.extend_from_slice(&self.bytes[cursor..end]);
            cursor = skip_ws(self.bytes, end);
            if self.bytes.get(cursor) != Some(&b')') {
                return Err(format!(
                    "unterminated :global(...) keyframes name in CSS module {}",
                    self.display
                ));
            }
            cursor += 1;
        } else {
            let end = ident_end(self.bytes, cursor);
            if end == cursor {
                return Err(format!(
                    "@keyframes without a name in CSS module {}",
                    self.display
                ));
            }
            let name = self.text[cursor..end].to_string();
            self.out
                .extend_from_slice(self.scoped(&name).as_bytes());
            cursor = end;
        }
        cursor = skip_ws_and_comments(self.bytes, cursor);
        if self.bytes.get(cursor) != Some(&b'{') {
            return Err(format!(
                "@keyframes without a body in CSS module {}",
                self.display
            ));
        }
        let close = find_matching(self.bytes, cursor, b'{', b'}', self.display)?;
        self.out.extend_from_slice(&self.bytes[cursor..close + 1]);
        self.pos = close + 1;
        Ok(())
    }

    fn parse_style_rule(&mut self) -> Result<(), String> {
        let start = self.pos;
        let mut cursor = self.pos;
        let brace = loop {
            if cursor >= self.bytes.len() {
                return Err(format!(
                    "expected `{{` after selector in CSS module {}",
                    self.display
                ));
            }
            match self.bytes[cursor] {
                b'{' => break cursor,
                b'}' | b';' => {
                    return Err(format!(
                        "cannot parse {:?} as a style rule in CSS module {}",
                        self.text[start..cursor + 1].trim(),
                        self.display
                    ));
                }
                b'"' | b'\'' => cursor = skip_string(self.bytes, cursor),
                b'/' if at_comment(self.bytes, cursor) => {
                    cursor = skip_comment(self.bytes, cursor);
                }
                b'(' | b'[' => {
                    let close = if self.bytes[cursor] == b'(' { b')' } else { b']' };
                    cursor =
                        find_matching(self.bytes, cursor, self.bytes[cursor], close, self.display)?
                            + 1;
                }
                _ => cursor += 1,
            }
        };
        let prelude = &self.text[start..brace];
        let scoped = self.scope_selector_part(prelude)?;
        self.out.extend_from_slice(scoped.as_bytes());
        self.out.push(b'{');
        let close = find_matching(self.bytes, brace, b'{', b'}', self.display)?;
        let body = &self.text[brace + 1..close];
        self.emit_declarations(prelude, body)?;
        self.out.push(b'}');
        self.pos = close + 1;
        Ok(())
    }

    /// Scopes one selector (or selector-list fragment): class/id locals are
    /// rewritten and registered, `:global(...)` contents pass through verbatim,
    /// `:local(...)` contents are scoped with the wrapper removed, and other
    /// pseudo-function arguments (`:not(...)`, `:is(...)`, ...) are scoped
    /// recursively. Anything the scoper cannot handle confidently is a hard
    /// error.
    fn scope_selector_part(&mut self, part: &str) -> Result<String, String> {
        let bytes = part.as_bytes();
        let mut out: Vec<u8> = Vec::with_capacity(part.len());
        let mut index = 0usize;
        while index < bytes.len() {
            match bytes[index] {
                b'"' | b'\'' => {
                    let end = skip_string(bytes, index);
                    out.extend_from_slice(&bytes[index..end]);
                    index = end;
                }
                b'/' if at_comment(bytes, index) => {
                    let end = skip_comment(bytes, index);
                    out.extend_from_slice(&bytes[index..end]);
                    index = end;
                }
                b'[' => {
                    let close = find_matching(bytes, index, b'[', b']', self.display)?;
                    out.extend_from_slice(&bytes[index..close + 1]);
                    index = close + 1;
                }
                b'\\' => {
                    return Err(format!(
                        "escaped selector characters are not supported in CSS module {} \
                         (selector {part:?})",
                        self.display
                    ));
                }
                b'&' => {
                    return Err(format!(
                        "CSS nesting (`&`) is not supported in CSS module {} \
                         (selector {part:?})",
                        self.display
                    ));
                }
                byte @ (b'.' | b'#') => {
                    let end = ident_end(bytes, index + 1);
                    let name = &part[index + 1..end];
                    if name.is_empty() || name.as_bytes()[0].is_ascii_digit() {
                        return Err(format!(
                            "cannot scope selector {part:?} in CSS module {}: `{}` is not \
                             followed by a valid name",
                            self.display,
                            char::from(byte)
                        ));
                    }
                    self.class_locals.entry(name.to_string()).or_default();
                    out.push(byte);
                    out.extend_from_slice(self.scoped(name).as_bytes());
                    index = end;
                }
                b':' => {
                    if bytes.get(index + 1) == Some(&b':') {
                        // Pseudo-element; a following functional argument
                        // (`::slotted(...)`) is scoped recursively.
                        let end = ident_end(bytes, index + 2);
                        out.extend_from_slice(&bytes[index..end]);
                        index = end;
                        index = self.scope_pseudo_arguments(part, index, &mut out)?;
                        continue;
                    }
                    let end = ident_end(bytes, index + 1);
                    let pseudo = part[index + 1..end].to_ascii_lowercase();
                    if pseudo == "global" || pseudo == "local" {
                        let open = skip_ws(bytes, end);
                        if bytes.get(open) != Some(&b'(') {
                            return Err(format!(
                                "bare `:{pseudo}` (without parentheses) is not supported in \
                                 CSS module {} (selector {part:?})",
                                self.display
                            ));
                        }
                        let close = find_matching(bytes, open, b'(', b')', self.display)?;
                        let inner = part[open + 1..close].trim();
                        if pseudo == "global" {
                            out.extend_from_slice(inner.as_bytes());
                        } else {
                            let scoped = self.scope_selector_part(inner)?;
                            out.extend_from_slice(scoped.as_bytes());
                        }
                        index = close + 1;
                    } else {
                        out.extend_from_slice(&bytes[index..end]);
                        index = end;
                        index = self.scope_pseudo_arguments(part, index, &mut out)?;
                    }
                }
                byte => {
                    out.push(byte);
                    index += 1;
                }
            }
        }
        String::from_utf8(out).map_err(|error| {
            format!(
                "scoped selector for CSS module {} is not valid UTF-8: {error}",
                self.display
            )
        })
    }

    /// If a `(` follows immediately, scopes the parenthesized pseudo arguments
    /// recursively and returns the index past the `)`; otherwise returns
    /// `index` unchanged.
    fn scope_pseudo_arguments(
        &mut self,
        part: &str,
        index: usize,
        out: &mut Vec<u8>,
    ) -> Result<usize, String> {
        let bytes = part.as_bytes();
        if bytes.get(index) != Some(&b'(') {
            return Ok(index);
        }
        let close = find_matching(bytes, index, b'(', b')', self.display)?;
        out.push(b'(');
        let scoped = self.scope_selector_part(&part[index + 1..close])?;
        out.extend_from_slice(scoped.as_bytes());
        out.push(b')');
        Ok(close + 1)
    }

    /// Emits a style rule's declarations: `composes:` is folded into the
    /// mapping (and dropped from the CSS), `animation`/`animation-name` values
    /// have local keyframes references rewritten, nested rules are a hard
    /// error, everything else is copied verbatim.
    fn emit_declarations(&mut self, prelude: &str, body: &str) -> Result<(), String> {
        let bytes = body.as_bytes();
        // Detect nested rules before splitting into declarations: a `{` in a
        // rule body (outside strings/comments) is CSS nesting.
        let mut probe = 0usize;
        while probe < bytes.len() {
            match bytes[probe] {
                b'"' | b'\'' => probe = skip_string(bytes, probe),
                b'/' if at_comment(bytes, probe) => probe = skip_comment(bytes, probe),
                b'{' => {
                    return Err(format!(
                        "nested rules are not supported in CSS module {} \
                         (in selector {:?})",
                        self.display,
                        prelude.trim()
                    ));
                }
                _ => probe += 1,
            }
        }
        for declaration in split_declarations(body) {
            let trimmed = declaration.trim();
            if trimmed.is_empty() {
                continue;
            }
            let Some(colon) = find_declaration_colon(trimmed) else {
                return Err(format!(
                    "cannot parse declaration {trimmed:?} in CSS module {}",
                    self.display
                ));
            };
            let property = trimmed[..colon].trim();
            let value = trimmed[colon + 1..].trim();
            let property_lower = property.to_ascii_lowercase();
            let base_property = property_lower
                .strip_prefix("-webkit-")
                .or_else(|| property_lower.strip_prefix("-moz-"))
                .or_else(|| property_lower.strip_prefix("-o-"))
                .unwrap_or(&property_lower);
            if base_property == "composes" {
                self.record_composes(prelude, value)?;
                continue;
            }
            if base_property == "animation" || base_property == "animation-name" {
                let rewritten = self.rewrite_animation_value(value);
                self.out
                    .extend_from_slice(format!("{property}: {rewritten};").as_bytes());
                continue;
            }
            self.out.extend_from_slice(trimmed.as_bytes());
            self.out.push(b';');
        }
        Ok(())
    }

    /// Parses one `composes:` declaration value and records the references on
    /// the rule's (single) class local.
    fn record_composes(&mut self, prelude: &str, value: &str) -> Result<(), String> {
        let selector = prelude.trim();
        let class_name = selector.strip_prefix('.').filter(|name| {
            !name.is_empty()
                && name.bytes().all(is_ident_byte)
                && !name.as_bytes()[0].is_ascii_digit()
        });
        let Some(class_name) = class_name else {
            return Err(format!(
                "`composes` is only valid inside a single-class selector rule \
                 (found {selector:?} in CSS module {})",
                self.display
            ));
        };
        let tokens = value.split_whitespace().collect::<Vec<_>>();
        let (names, source) = match tokens.iter().position(|token| *token == "from") {
            Some(position) => (&tokens[..position], Some(&tokens[position + 1..])),
            None => (&tokens[..], None),
        };
        if names.is_empty() {
            return Err(format!(
                "`composes` with no class names ({value:?}) in CSS module {}",
                self.display
            ));
        }
        for name in names {
            if name.is_empty()
                || !name.bytes().all(is_ident_byte)
                || name.as_bytes()[0].is_ascii_digit()
            {
                return Err(format!(
                    "`composes` target {name:?} is not a valid class name in CSS module {}",
                    self.display
                ));
            }
        }
        let references: Vec<ComposeRef> = match source {
            None => names
                .iter()
                .map(|name| ComposeRef::Local((*name).to_string()))
                .collect(),
            Some([source]) if *source == "global" => names
                .iter()
                .map(|name| ComposeRef::Global((*name).to_string()))
                .collect(),
            Some([source])
                if source.len() >= 2
                    && (source.starts_with('"') && source.ends_with('"')
                        || source.starts_with('\'') && source.ends_with('\'')) =>
            {
                let specifier = &source[1..source.len() - 1];
                if specifier.is_empty() {
                    return Err(format!(
                        "`composes ... from` with an empty source in CSS module {}",
                        self.display
                    ));
                }
                names
                    .iter()
                    .map(|name| ComposeRef::Foreign {
                        specifier: specifier.to_string(),
                        name: (*name).to_string(),
                    })
                    .collect()
            }
            Some(source) => {
                return Err(format!(
                    "unsupported `composes ... from {}` in CSS module {}; expected a quoted \
                     file or `global`",
                    source.join(" "),
                    self.display
                ));
            }
        };
        self.class_locals
            .entry(class_name.to_string())
            .or_default()
            .extend(references);
        Ok(())
    }

    /// Rewrites tokens in an `animation`/`animation-name` value that exactly
    /// match a locally-declared `@keyframes` name to the scoped name. Function
    /// groups (`var(...)`, `cubic-bezier(...)`) are copied whole and never
    /// rewritten.
    fn rewrite_animation_value(&self, value: &str) -> String {
        let bytes = value.as_bytes();
        let mut out = String::with_capacity(value.len());
        let mut index = 0usize;
        while index < bytes.len() {
            match bytes[index] {
                b'"' | b'\'' => {
                    let end = skip_string(bytes, index);
                    let token = &value[index + 1..end.saturating_sub(1).max(index + 1)];
                    if self.keyframes_names.contains(token) {
                        out.push(char::from(bytes[index]));
                        out.push_str(&self.scoped(token));
                        out.push(char::from(bytes[index]));
                    } else {
                        out.push_str(&value[index..end]);
                    }
                    index = end;
                }
                byte if is_ident_byte(byte) => {
                    let end = ident_end(bytes, index);
                    let token = &value[index..end];
                    if bytes.get(end) == Some(&b'(') {
                        // A function: copy it (and its arguments) verbatim.
                        let close = find_matching(bytes, end, b'(', b')', self.display)
                            .unwrap_or(bytes.len().saturating_sub(1));
                        out.push_str(&value[index..(close + 1).min(value.len())]);
                        index = (close + 1).min(value.len());
                    } else {
                        if self.keyframes_names.contains(token) {
                            out.push_str(&self.scoped(token));
                        } else {
                            out.push_str(token);
                        }
                        index = end;
                    }
                }
                byte => {
                    out.push(char::from(byte));
                    index += 1;
                }
            }
        }
        out
    }

    /// Builds the deterministic export mapping: every class/id local and every
    /// scoped keyframes name, sorted by name, with same-file `composes` chains
    /// resolved transitively (a cycle is a hard error) and cross-file composes
    /// recorded as runtime lookups.
    fn resolve_mapping(&self) -> Result<(CssMapping, Vec<String>), String> {
        let mut compose_imports: Vec<String> = Vec::new();
        let mut mapping = Vec::new();
        let mut names: BTreeSet<&String> = self.class_locals.keys().collect();
        names.extend(self.keyframes_names.iter());
        for name in names {
            let mut segments = Vec::new();
            let mut stack = Vec::new();
            self.value_of(name, &mut segments, &mut stack, &mut compose_imports)?;
            mapping.push((name.clone(), segments));
        }
        Ok((mapping, compose_imports))
    }

    fn value_of<'n>(
        &'n self,
        name: &'n str,
        segments: &mut Vec<MappingSegment>,
        stack: &mut Vec<&'n str>,
        compose_imports: &mut Vec<String>,
    ) -> Result<(), String> {
        if stack.contains(&name) {
            return Err(format!(
                "circular `composes` chain involving {name:?} in CSS module {}",
                self.display
            ));
        }
        stack.push(name);
        segments.push(MappingSegment::Literal(self.scoped(name)));
        for reference in self.class_locals.get(name).into_iter().flatten() {
            match reference {
                ComposeRef::Local(other) => {
                    if !self.class_locals.contains_key(other) {
                        return Err(format!(
                            "`composes` target {other:?} is not defined as a class in \
                             CSS module {}",
                            self.display
                        ));
                    }
                    self.value_of(other, segments, stack, compose_imports)?;
                }
                ComposeRef::Global(global) => {
                    segments.push(MappingSegment::Literal(global.clone()));
                }
                ComposeRef::Foreign { specifier, name } => {
                    let import = match compose_imports
                        .iter()
                        .position(|existing| existing == specifier)
                    {
                        Some(existing) => existing,
                        None => {
                            compose_imports.push(specifier.clone());
                            compose_imports.len() - 1
                        }
                    };
                    segments.push(MappingSegment::Foreign {
                        import,
                        name: name.clone(),
                    });
                }
            }
        }
        stack.pop();
        Ok(())
    }
}

/// Splits a rule body into declarations on top-level `;` (strings, comments,
/// and parenthesized groups are honoured).
fn split_declarations(body: &str) -> Vec<&str> {
    let bytes = body.as_bytes();
    let mut declarations = Vec::new();
    let mut start = 0usize;
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'"' | b'\'' => index = skip_string(bytes, index),
            b'/' if at_comment(bytes, index) => index = skip_comment(bytes, index),
            b'(' => {
                index = match find_matching(bytes, index, b'(', b')', "declaration") {
                    Ok(close) => close + 1,
                    Err(_) => bytes.len(),
                }
            }
            b';' => {
                declarations.push(&body[start..index]);
                index += 1;
                start = index;
            }
            _ => index += 1,
        }
    }
    if start < body.len() {
        declarations.push(&body[start..]);
    }
    declarations
}

/// The index of the `:` separating property from value (skipping strings and
/// parenthesized groups), or `None`.
fn find_declaration_colon(declaration: &str) -> Option<usize> {
    let bytes = declaration.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'"' | b'\'' => index = skip_string(bytes, index),
            b'(' => {
                index = match find_matching(bytes, index, b'(', b')', "declaration") {
                    Ok(close) => close + 1,
                    Err(_) => bytes.len(),
                }
            }
            b':' => return Some(index),
            _ => index += 1,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module(css: &str) -> ProcessedCss {
        process_css_module(Path::new("/project/button.module.css"), css)
            .expect("module should process")
    }

    fn module_error(css: &str) -> String {
        match process_css_module(Path::new("/project/button.module.css"), css) {
            Ok(_) => panic!("expected a hard error for {css:?}"),
            Err(error) => error,
        }
    }

    fn scoped_of(processed: &ProcessedCss, name: &str) -> String {
        let (_, segments) = processed
            .mapping
            .iter()
            .find(|(local, _)| local == name)
            .unwrap_or_else(|| panic!("no mapping for {name}"));
        match &segments[0] {
            MappingSegment::Literal(literal) => literal.clone(),
            MappingSegment::Foreign { .. } => panic!("first segment must be the scoped self"),
        }
    }

    #[test]
    fn scopes_classes_across_combinators_pseudos_and_comma_lists() {
        let processed = module(
            ".foo { color: red; }\n\
             .foo:hover, .bar::before { color: blue; }\n\
             .foo > .bar + .baz ~ .qux { color: green; }\n\
             div.foo[data-x=\".raw\"] { color: black; }\n",
        );
        let foo = scoped_of(&processed, "foo");
        assert!(foo.starts_with("_foo_"), "{foo}");
        assert!(processed.css.contains(&format!(".{foo}:hover")), "{}", processed.css);
        let bar = scoped_of(&processed, "bar");
        assert!(processed.css.contains(&format!(".{bar}::before")), "{}", processed.css);
        assert!(
            processed.css.contains(&format!(".{foo} > .{bar}")),
            "{}",
            processed.css
        );
        // The attribute string is untouched; the type selector keeps its class.
        assert!(processed.css.contains("[data-x=\".raw\"]"), "{}", processed.css);
        assert!(!processed.css.contains(".foo"), "no unscoped local: {}", processed.css);
        // Every local appears in the mapping.
        for name in ["foo", "bar", "baz", "qux"] {
            assert!(scoped_of(&processed, name).starts_with(&format!("_{name}_")));
        }
    }

    #[test]
    fn the_scoped_name_is_stable_for_unchanged_content_and_moves_with_content() {
        let file = Path::new("/project/a.module.css");
        let first = process_css_module(file, ".foo { color: red; }").unwrap();
        let second = process_css_module(file, ".foo { color: red; }").unwrap();
        assert_eq!(first.mapping, second.mapping, "stable across rebuilds");
        let changed = process_css_module(file, ".foo { color: blue; }").unwrap();
        assert_ne!(
            scoped_of(&first, "foo"),
            scoped_of(&changed, "foo"),
            "content change moves the hash"
        );
    }

    #[test]
    fn global_leaves_contents_unscoped_and_local_scopes_explicitly() {
        let processed = module(
            ":global(.raw) .item { color: red; }\n\
             :global(.a .b) { color: blue; }\n\
             :local(.mine) { color: green; }\n",
        );
        let item = scoped_of(&processed, "item");
        let mine = scoped_of(&processed, "mine");
        assert!(processed.css.contains(&format!(".raw .{item}")), "{}", processed.css);
        assert!(processed.css.contains(".a .b"), "{}", processed.css);
        assert!(processed.css.contains(&format!(".{mine}")), "{}", processed.css);
        assert!(!processed.css.contains(":global"), "{}", processed.css);
        assert!(!processed.css.contains(":local"), "{}", processed.css);
        // Global names are not exported.
        assert!(!processed.mapping.iter().any(|(name, _)| name == "raw"));
    }

    #[test]
    fn classes_inside_selector_pseudo_functions_are_scoped() {
        let processed = module(".foo:not(.bar):is(.baz) { color: red; }");
        let bar = scoped_of(&processed, "bar");
        let baz = scoped_of(&processed, "baz");
        assert!(processed.css.contains(&format!(":not(.{bar})")), "{}", processed.css);
        assert!(processed.css.contains(&format!(":is(.{baz})")), "{}", processed.css);
    }

    #[test]
    fn same_file_composes_chains_resolve_transitively_in_order() {
        let processed = module(
            ".base { color: red; }\n\
             .mid { composes: base; color: green; }\n\
             .top { composes: mid; color: blue; }\n",
        );
        let (_, segments) = processed
            .mapping
            .iter()
            .find(|(name, _)| name == "top")
            .unwrap();
        let literals = segments
            .iter()
            .map(|segment| match segment {
                MappingSegment::Literal(literal) => literal.clone(),
                MappingSegment::Foreign { .. } => panic!("no foreign segments here"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            literals,
            vec![
                scoped_of(&processed, "top"),
                scoped_of(&processed, "mid"),
                scoped_of(&processed, "base")
            ]
        );
        assert!(!processed.css.contains("composes"), "{}", processed.css);
    }

    #[test]
    fn composes_from_a_file_and_from_global_produce_the_right_segments() {
        let processed = module(
            ".foo { composes: bar baz from './other.module.css'; composes: raw from global; }",
        );
        assert_eq!(processed.compose_imports, vec!["./other.module.css".to_string()]);
        let (_, segments) = processed
            .mapping
            .iter()
            .find(|(name, _)| name == "foo")
            .unwrap();
        assert_eq!(segments.len(), 4);
        assert!(matches!(&segments[1], MappingSegment::Foreign { import: 0, name } if name == "bar"));
        assert!(matches!(&segments[2], MappingSegment::Foreign { import: 0, name } if name == "baz"));
        assert!(matches!(&segments[3], MappingSegment::Literal(literal) if literal == "raw"));
    }

    #[test]
    fn keyframes_names_and_animation_references_are_scoped_together() {
        let processed = module(
            ".spinner { animation: spin 1s linear infinite, var(--other); }\n\
             .named { animation-name: spin; }\n\
             @keyframes spin { from { transform: rotate(0); } to { transform: rotate(360deg); } }\n\
             @keyframes :global(shared) { from { opacity: 0; } }\n",
        );
        let spin = scoped_of(&processed, "spin");
        assert!(spin.starts_with("_spin_"), "{spin}");
        assert!(processed.css.contains(&format!("@keyframes {spin}")), "{}", processed.css);
        assert!(processed.css.contains(&format!("animation: {spin} 1s")), "{}", processed.css);
        assert!(
            processed.css.contains(&format!("animation-name: {spin};")),
            "{}",
            processed.css
        );
        assert!(processed.css.contains("var(--other)"), "{}", processed.css);
        assert!(processed.css.contains("@keyframes shared"), "{}", processed.css);
    }

    #[test]
    fn conditional_group_rules_scope_their_nested_style_rules() {
        let processed = module(
            "@media (min-width: 600px) { .wide { color: red; } }\n\
             @supports (display: grid) { .grid { display: grid; } }\n",
        );
        let wide = scoped_of(&processed, "wide");
        assert!(processed.css.contains("@media (min-width: 600px)"), "{}", processed.css);
        assert!(processed.css.contains(&format!(".{wide}")), "{}", processed.css);
        assert!(!processed.css.contains(".wide"), "{}", processed.css);
    }

    #[test]
    fn unsupported_constructs_are_hard_errors_naming_the_file_and_construct() {
        let unknown_at_rule = module_error("@tailwind base;\n.foo { color: red; }");
        assert!(unknown_at_rule.contains("unsupported at-rule `@tailwind`"), "{unknown_at_rule}");
        assert!(unknown_at_rule.contains("button.module.css"), "{unknown_at_rule}");

        let nested = module_error(".foo { .bar { color: red; } }");
        assert!(nested.contains("nested rules are not supported"), "{nested}");

        let bare_global = module_error(":global .foo { color: red; }");
        assert!(bare_global.contains("bare `:global`"), "{bare_global}");

        let escaped = module_error(".foo\\:hover { color: red; }");
        assert!(escaped.contains("escaped selector"), "{escaped}");

        let compound_composes = module_error(".a .b { composes: c; }");
        assert!(
            compound_composes.contains("`composes` is only valid inside a single-class"),
            "{compound_composes}"
        );

        let unknown_target = module_error(".a { composes: missing; }");
        assert!(
            unknown_target.contains("`composes` target \"missing\" is not defined"),
            "{unknown_target}"
        );

        let cycle = module_error(".a { composes: b; }\n.b { composes: a; }");
        assert!(cycle.contains("circular `composes` chain"), "{cycle}");

        let nesting = module_error(".foo { color: red; } .bar & { color: blue; }");
        assert!(nesting.contains("CSS nesting"), "{nesting}");
    }

    #[test]
    fn import_statements_parse_string_and_url_forms_with_media() {
        let file = Path::new("/project/app.css");
        let processed = process_global_css(
            file,
            "@import './a.css';\n\
             @import url(\"./b.css\");\n\
             @import './c.css' screen and (min-width: 600px);\n\
             @import url(https://example.com/font.css);\n\
             .app { color: red; }\n",
        )
        .unwrap();
        assert_eq!(
            processed.imports,
            vec![
                CssImport { specifier: "./a.css".into(), media: None },
                CssImport { specifier: "./b.css".into(), media: None },
                CssImport {
                    specifier: "./c.css".into(),
                    media: Some("screen and (min-width: 600px)".into())
                },
            ]
        );
        assert_eq!(
            processed.external_imports,
            vec!["@import url(https://example.com/font.css);".to_string()]
        );
        assert!(!processed.css.contains("@import"), "{}", processed.css);
        assert!(processed.css.contains(".app { color: red; }"), "{}", processed.css);
    }

    #[test]
    fn an_import_layer_or_supports_condition_is_a_hard_error() {
        let file = Path::new("/project/app.css");
        let layer = process_global_css(file, "@import './a.css' layer(base);").unwrap_err();
        assert!(layer.contains("layer(...) condition is not supported"), "{layer}");
        assert!(layer.contains("app.css"), "{layer}");
        let supports =
            process_global_css(file, "@import './a.css' supports(display: grid);").unwrap_err();
        assert!(supports.contains("supports(...) condition is not supported"), "{supports}");
    }

    #[test]
    fn url_rewriting_skips_absolute_data_fragment_and_public_urls() {
        let file = Path::new("/project/app.css");
        let processed = process_global_css(
            file,
            ".a { background: url(https://example.com/x.png); }\n\
             .b { background: url(data:image/png;base64,AAAA); }\n\
             .c { fill: url(#gradient); }\n\
             .d { background: url(/public/logo.png); }\n\
             .e { background: url(//cdn.example.com/x.png); }\n\
             /* url(./commented.png) */\n\
             .f { content: \"url(./quoted.png)\"; }\n",
        )
        .unwrap();
        assert!(processed.assets.is_empty(), "{:?}", processed.assets);
        for original in [
            "url(https://example.com/x.png)",
            "url(data:image/png;base64,AAAA)",
            "url(#gradient)",
            "url(/public/logo.png)",
            "url(//cdn.example.com/x.png)",
            "url(./commented.png)",
            "\"url(./quoted.png)\"",
        ] {
            assert!(processed.css.contains(original), "{original} must survive: {}", processed.css);
        }
    }

    #[test]
    fn a_missing_url_target_names_the_css_file_and_the_reference() {
        let error = process_global_css(
            Path::new("/project/app.css"),
            ".a { background: url(./missing.png); }",
        )
        .unwrap_err();
        assert!(error.contains("url(./missing.png)"), "{error}");
        assert!(error.contains("/project/app.css"), "{error}");
        assert!(error.contains("missing.png"), "{error}");
    }

    #[test]
    fn recognizes_css_module_paths() {
        assert!(is_css_module_path(Path::new("/a/button.module.css")));
        assert!(!is_css_module_path(Path::new("/a/button.css")));
        assert!(!is_css_module_path(Path::new("/a/.module.css")));
        assert!(!is_css_module_path(Path::new("/a/module.css")));
    }
}
