//! Native SCSS compilation (subset).
//!
//! Compiles `.scss` sources to plain CSS *before* the existing CSS pipeline
//! runs, so a Sass file flows through the same global-stylesheet /
//! CSS-Modules loaders as a hand-written `.css` file. Owned code, no `grass`,
//! in the same spirit as [`crate::tailwind`] and [`crate::css`].
//!
//! Supported (driven by real-world Vite apps, implemented as general
//! patterns):
//!
//! - **Variables**: `$x: value;` declarations (file-, rule-, and mixin-local
//!   scopes), references in any value position, `!default` (assign-if-unset).
//! - **Nesting** to any depth with `&` in every selector position
//!   (`&:hover`, `&.mod`, `.parent &`), selector lists cross-multiplied, and
//!   nested `@media`/`@supports` bubbling to the top level.
//! - **`@mixin` / `@include`** with positional arguments and defaults,
//!   including mixins that contain nested rules and other includes.
//! - **`@use 'file' as ns;`** (namespace member access `ns.$var`,
//!   `@include ns.mixin`), the default namespace (`@use './x';` -> `x.$var`),
//!   `as *`, root-relative `/src/...` targets (Vite semantics), and the
//!   `_partial.scss` naming convention. Each file is loaded once per
//!   compilation; its CSS is emitted at the first `@use`.
//! - **`@import './x'`** of a relative Sass partial: evaluated in place in the
//!   importer's scope (Sass import semantics). Plain-CSS imports (`.css`,
//!   `url(...)`, remote) pass through for the downstream CSS pipeline.
//! - **Expressions**: arithmetic on numbers with units (`2 * $padding`,
//!   `-$size * 3`), `sqrt(...)` of unitless numbers, and full `calc(...)`
//!   simplification with dart-sass semantics (same-unit terms combine,
//!   `calc(65vmin / 2)` -> `32.5vmin`, mixed units stay symbolic:
//!   `calc(50vh - 32.5vmin)`).
//!
//! Everything else is a **hard error naming the file and the construct** —
//! `@extend`, control flow (`@if`/`@each`/`@for`/...), `@function`,
//! interpolation `#{...}`, placeholder selectors, `@use ... with (...)`,
//!   Sass-only color/list/map functions (`darken`, `map-get`, ...),
//! namespaced function calls (`math.div`), keyword arguments, and the
//! indented `.sass` syntax. Never silently wrong output.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use crate::css::{
    at_comment, find_matching, ident_end, is_ident_byte, skip_string, skip_ws, starts_with_ci,
};

/// Options threaded from the build configuration.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScssOptions {
    /// Vite's `css.preprocessorOptions.scss.additionalData` when it is a
    /// string: prepended to every compiled root file (not to `@use`d files).
    pub additional_data: Option<String>,
    /// The project root, for resolving root-relative `@use "/src/..."`
    /// targets (Vite resolves them against the project root).
    pub root: Option<PathBuf>,
}

/// The result of compiling one root `.scss` file.
#[derive(Debug, Clone)]
pub struct CompiledScss {
    /// Plain CSS, ready for [`crate::css::process_global_css`] or
    /// [`crate::css::process_css_module`].
    pub css: String,
    /// Every OTHER physical file whose content contributed (`@use`/`@import`
    /// targets, transitively). The caller records these so edits invalidate.
    pub loaded_files: Vec<PathBuf>,
}

/// Whether a resolved path is a Sass source (`.scss` or `.sass`).
pub fn is_scss_path(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|value| value.to_str()),
        Some("scss" | "sass")
    )
}

/// Whether a resolved path is a Sass CSS Module (`*.module.scss`).
pub fn is_scss_module_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".module.scss") && name != ".module.scss")
}

/// Compiles one root `.scss` file to plain CSS. `additional_data` (when
/// configured) is prepended to the source, Vite-style.
pub fn compile_scss(
    file: &Path,
    source: &str,
    options: &ScssOptions,
) -> Result<CompiledScss, String> {
    if file.extension().and_then(|value| value.to_str()) == Some("sass") {
        return Err(format!(
            "{}: the indented `.sass` syntax is not supported (construct: indented syntax); \
             use `.scss`",
            file.display()
        ));
    }
    let mut text;
    let source = match &options.additional_data {
        Some(additional) => {
            text = String::with_capacity(additional.len() + 1 + source.len());
            text.push_str(additional);
            text.push('\n');
            text.push_str(source);
            text.as_str()
        }
        None => source,
    };
    let display = file.display().to_string();
    let clean = strip_sass_comments(source);
    let items = parse_block_str(&clean, &display)?;
    let mut compiler = Compiler {
        options,
        root_file: file.to_path_buf(),
        modules: HashMap::new(),
        loading: Vec::new(),
        loaded_files: Vec::new(),
    };
    let mut ctx = FileCtx::new(file.to_path_buf());
    let mut out = Vec::new();
    compiler.eval_top_level(&items, &mut ctx, &mut out)?;
    let mut css = String::new();
    render_nodes(&out, 0, &mut css);
    Ok(CompiledScss {
        css,
        loaded_files: compiler.loaded_files,
    })
}

// ---------------------------------------------------------------------------
// Comment stripping (`//` silent comments and `/* */` block comments).
// ---------------------------------------------------------------------------

/// Removes both comment forms, preserving strings and (possibly unquoted)
/// `url(...)` contents — a base64 data URI legally contains `//`.
fn strip_sass_comments(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'"' | b'\'' => {
                let end = skip_string(bytes, index);
                out.extend_from_slice(&bytes[index..end]);
                index = end;
            }
            b'/' if bytes.get(index + 1) == Some(&b'/') => {
                while index < bytes.len() && bytes[index] != b'\n' {
                    index += 1;
                }
            }
            b'/' if at_comment(bytes, index) => {
                let mut cursor = index + 2;
                while cursor + 1 < bytes.len()
                    && !(bytes[cursor] == b'*' && bytes[cursor + 1] == b'/')
                {
                    cursor += 1;
                }
                index = (cursor + 2).min(bytes.len());
            }
            b'u' | b'U'
                if starts_with_ci(bytes, index, "url(")
                    && (index == 0 || !is_ident_byte(bytes[index - 1])) =>
            {
                let end = match find_matching(bytes, index + 3, b'(', b')', "url") {
                    Ok(close) => close + 1,
                    Err(_) => bytes.len(),
                };
                out.extend_from_slice(&bytes[index..end]);
                index = end;
            }
            byte => {
                out.push(byte);
                index += 1;
            }
        }
    }
    String::from_utf8(out).unwrap_or_else(|_| source.to_string())
}

// ---------------------------------------------------------------------------
// Parser: source text -> item tree.
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum Item {
    VarDecl {
        name: String,
        value: String,
        default: bool,
    },
    Declaration {
        property: String,
        value: String,
    },
    Rule {
        selector: String,
        body: Vec<Item>,
    },
    /// `@media`, `@supports`, `@container`, `@font-face`, `@page`,
    /// `@keyframes` (and vendor-prefixed), block `@layer`.
    AtBlock {
        name: String,
        prelude: String,
        body: Vec<Item>,
    },
    Use {
        path: String,
        namespace: UseNamespace,
    },
    Import {
        target: String,
    },
    /// A plain-CSS `@import` (remote, `.css`, `url(...)`, media-qualified):
    /// reproduced verbatim for the downstream CSS pipeline.
    RawImport {
        statement: String,
    },
    MixinDecl {
        name: String,
        params: Vec<(String, Option<String>)>,
        body: Rc<Vec<Item>>,
    },
    Include {
        target: String,
        args: Vec<String>,
    },
    /// `@layer a, b;` — passed through verbatim.
    RawStatement {
        statement: String,
    },
}

#[derive(Debug, Clone)]
enum UseNamespace {
    Default,
    Named(String),
    Star,
}

fn parse_block_str(text: &str, file: &str) -> Result<Vec<Item>, String> {
    let bytes = text.as_bytes();
    let mut pos = 0usize;
    let items = parse_block(text, bytes, &mut pos, file, true)?;
    Ok(items)
}

/// Constructs that are recognized by name and rejected loudly.
const UNSUPPORTED_AT_RULES: &[&str] = &[
    "extend", "if", "else", "each", "for", "while", "function", "return", "forward", "at-root",
    "debug", "warn", "error", "content",
];

fn parse_block(
    text: &str,
    bytes: &[u8],
    pos: &mut usize,
    file: &str,
    top: bool,
) -> Result<Vec<Item>, String> {
    let mut items = Vec::new();
    loop {
        *pos = skip_ws(bytes, *pos);
        if *pos >= bytes.len() {
            if top {
                return Ok(items);
            }
            return Err(format!("{file}: unterminated block"));
        }
        match bytes[*pos] {
            b'}' => {
                if top {
                    return Err(format!("{file}: unmatched `}}`"));
                }
                *pos += 1;
                return Ok(items);
            }
            b'$' => items.push(parse_var_decl(text, bytes, pos, file)?),
            b'@' => {
                if let Some(item) = parse_at_rule(text, bytes, pos, file, top)? {
                    items.push(item);
                }
            }
            _ => items.push(parse_rule_or_declaration(text, bytes, pos, file)?),
        }
    }
}

/// Scans forward (string/paren/bracket aware) for the first top-level `{`,
/// `;`, or `}`. Returns `(index, byte_found)`; `byte_found` is `0` at EOF.
fn scan_item_end(bytes: &[u8], start: usize, file: &str) -> Result<(usize, u8), String> {
    let mut cursor = start;
    while cursor < bytes.len() {
        match bytes[cursor] {
            b'"' | b'\'' => cursor = skip_string(bytes, cursor),
            b'(' => cursor = find_matching(bytes, cursor, b'(', b')', file)? + 1,
            b'[' => cursor = find_matching(bytes, cursor, b'[', b']', file)? + 1,
            b'#' if bytes.get(cursor + 1) == Some(&b'{') => {
                return Err(format!(
                    "{file}: interpolation `#{{...}}` is not supported"
                ));
            }
            byte @ (b'{' | b';' | b'}') => return Ok((cursor, byte)),
            _ => cursor += 1,
        }
    }
    Ok((bytes.len(), 0))
}

fn check_no_interpolation(fragment: &str, file: &str) -> Result<(), String> {
    if let Some(position) = fragment.find("#{") {
        // Inside a quoted string interpolation would still be Sass; but a
        // `#{` in a string is overwhelmingly real interpolation intent.
        let _ = position;
        return Err(format!(
            "{file}: interpolation `#{{...}}` is not supported (in {fragment:?})"
        ));
    }
    Ok(())
}

fn parse_var_decl(
    text: &str,
    bytes: &[u8],
    pos: &mut usize,
    file: &str,
) -> Result<Item, String> {
    let name_start = *pos + 1;
    let name_end = ident_end(bytes, name_start);
    if name_end == name_start {
        return Err(format!("{file}: `$` is not followed by a variable name"));
    }
    let name = text[name_start..name_end].to_string();
    let mut cursor = skip_ws(bytes, name_end);
    if bytes.get(cursor) != Some(&b':') {
        return Err(format!(
            "{file}: expected `:` after variable `${name}`"
        ));
    }
    cursor += 1;
    let (end, found) = scan_item_end(bytes, cursor, file)?;
    if found == b'{' {
        return Err(format!(
            "{file}: cannot parse the value of `${name}` (unexpected `{{`)"
        ));
    }
    let mut value = text[cursor..end].trim().to_string();
    check_no_interpolation(&value, file)?;
    let mut default = false;
    loop {
        if let Some(stripped) = value.strip_suffix("!default") {
            value = stripped.trim_end().to_string();
            default = true;
        } else if value.ends_with("!global") {
            return Err(format!(
                "{file}: `!global` on `${name}` is not supported"
            ));
        } else {
            break;
        }
    }
    *pos = if found == b';' { end + 1 } else { end };
    Ok(Item::VarDecl {
        name,
        value,
        default,
    })
}

fn parse_at_rule(
    text: &str,
    bytes: &[u8],
    pos: &mut usize,
    file: &str,
    top: bool,
) -> Result<Option<Item>, String> {
    let at = *pos;
    let name_end = ident_end(bytes, at + 1);
    let raw_name = &text[at + 1..name_end];
    let name = raw_name.to_ascii_lowercase();
    if UNSUPPORTED_AT_RULES.contains(&name.as_str()) {
        return Err(format!(
            "{file}: `@{raw_name}` is not supported (construct: @{raw_name})"
        ));
    }
    match name.as_str() {
        "use" => {
            if !top {
                return Err(format!("{file}: `@use` is only allowed at the top level"));
            }
            let (end, found) = scan_item_end(bytes, name_end, file)?;
            if found == b'{' {
                return Err(format!("{file}: cannot parse `@use` (unexpected `{{`)"));
            }
            let body = text[name_end..end].trim();
            *pos = if found == b';' { end + 1 } else { end };
            Ok(Some(parse_use(body, file)?))
        }
        "import" => {
            let (end, found) = scan_item_end(bytes, name_end, file)?;
            if found == b'{' {
                return Err(format!("{file}: cannot parse `@import` (unexpected `{{`)"));
            }
            let body = text[name_end..end].trim();
            let statement = format!("@import {body};");
            *pos = if found == b';' { end + 1 } else { end };
            let single_quoted = body.len() >= 2
                && (body.starts_with('"') && body.ends_with('"')
                    || body.starts_with('\'') && body.ends_with('\''));
            if single_quoted {
                let target = body[1..body.len() - 1].to_string();
                let plain_css = target.starts_with("http:")
                    || target.starts_with("https:")
                    || target.starts_with("//")
                    || target.ends_with(".css");
                if plain_css {
                    Ok(Some(Item::RawImport { statement }))
                } else {
                    Ok(Some(Item::Import { target }))
                }
            } else {
                // `url(...)`, comma lists, media-qualified: plain-CSS import.
                Ok(Some(Item::RawImport { statement }))
            }
        }
        "charset" => {
            let (end, found) = scan_item_end(bytes, name_end, file)?;
            *pos = if found == b';' { end + 1 } else { end };
            Ok(None)
        }
        "mixin" => {
            let cursor = skip_ws(bytes, name_end);
            let mixin_end = ident_end(bytes, cursor);
            if mixin_end == cursor {
                return Err(format!("{file}: `@mixin` without a name"));
            }
            let mixin_name = text[cursor..mixin_end].to_string();
            let mut cursor = skip_ws(bytes, mixin_end);
            let params = if bytes.get(cursor) == Some(&b'(') {
                let close = find_matching(bytes, cursor, b'(', b')', file)?;
                let raw = &text[cursor + 1..close];
                cursor = skip_ws(bytes, close + 1);
                parse_mixin_params(raw, &mixin_name, file)?
            } else {
                Vec::new()
            };
            if bytes.get(cursor) != Some(&b'{') {
                return Err(format!(
                    "{file}: expected `{{` after `@mixin {mixin_name}`"
                ));
            }
            *pos = cursor + 1;
            let body = parse_block(text, bytes, pos, file, false)?;
            Ok(Some(Item::MixinDecl {
                name: mixin_name,
                params,
                body: Rc::new(body),
            }))
        }
        "include" => {
            let cursor = skip_ws(bytes, name_end);
            let mut target_end = ident_end(bytes, cursor);
            if target_end == cursor {
                return Err(format!("{file}: `@include` without a mixin name"));
            }
            if bytes.get(target_end) == Some(&b'.') {
                let member_end = ident_end(bytes, target_end + 1);
                if member_end == target_end + 1 {
                    return Err(format!(
                        "{file}: `@include {}` has an empty member name",
                        &text[cursor..target_end + 1]
                    ));
                }
                target_end = member_end;
            }
            let target = text[cursor..target_end].to_string();
            let mut cursor = skip_ws(bytes, target_end);
            let args = if bytes.get(cursor) == Some(&b'(') {
                let close = find_matching(bytes, cursor, b'(', b')', file)?;
                let raw = &text[cursor + 1..close];
                cursor = skip_ws(bytes, close + 1);
                split_top_level_commas(raw)
                    .into_iter()
                    .map(|argument| argument.trim().to_string())
                    .filter(|argument| !argument.is_empty())
                    .collect()
            } else {
                Vec::new()
            };
            if bytes.get(cursor) == Some(&b'{') {
                return Err(format!(
                    "{file}: `@include {target} {{ ... }}` content blocks are not supported \
                     (construct: @content block)"
                ));
            }
            if bytes.get(cursor) == Some(&b';') {
                cursor += 1;
            }
            *pos = cursor;
            Ok(Some(Item::Include { target, args }))
        }
        "media" | "supports" | "container" | "font-face" | "page" => {
            let (brace, found) = scan_item_end(bytes, name_end, file)?;
            if found != b'{' {
                return Err(format!("{file}: `@{raw_name}` without a block"));
            }
            let prelude = text[name_end..brace].trim().to_string();
            check_no_interpolation(&prelude, file)?;
            *pos = brace + 1;
            let body = parse_block(text, bytes, pos, file, false)?;
            Ok(Some(Item::AtBlock {
                name,
                prelude,
                body,
            }))
        }
        "layer" => {
            let (end, found) = scan_item_end(bytes, name_end, file)?;
            match found {
                b';' => {
                    let statement = text[at..end + 1].to_string();
                    *pos = end + 1;
                    Ok(Some(Item::RawStatement { statement }))
                }
                b'{' => {
                    let prelude = text[name_end..end].trim().to_string();
                    *pos = end + 1;
                    let body = parse_block(text, bytes, pos, file, false)?;
                    Ok(Some(Item::AtBlock {
                        name,
                        prelude,
                        body,
                    }))
                }
                _ => Err(format!("{file}: cannot parse `@layer`")),
            }
        }
        _ if name == "keyframes" || name.ends_with("-keyframes") => {
            let (brace, found) = scan_item_end(bytes, name_end, file)?;
            if found != b'{' {
                return Err(format!("{file}: `@{raw_name}` without a block"));
            }
            let prelude = text[name_end..brace].trim().to_string();
            check_no_interpolation(&prelude, file)?;
            *pos = brace + 1;
            let body = parse_block(text, bytes, pos, file, false)?;
            Ok(Some(Item::AtBlock {
                name: raw_name.to_string(),
                prelude,
                body,
            }))
        }
        _ => Err(format!(
            "{file}: unsupported at-rule `@{raw_name}` (construct: @{raw_name})"
        )),
    }
}

fn parse_use(body: &str, file: &str) -> Result<Item, String> {
    let body = body.trim();
    let (path, rest) = if body.len() >= 2 && (body.starts_with('"') || body.starts_with('\'')) {
        let quote = body.as_bytes()[0];
        let end = skip_string(body.as_bytes(), 0);
        if end < 2 || body.as_bytes().get(end - 1) != Some(&quote) {
            return Err(format!("{file}: unterminated `@use` path in {body:?}"));
        }
        (body[1..end - 1].to_string(), body[end..].trim())
    } else {
        return Err(format!(
            "{file}: `@use` expects a quoted path (found {body:?})"
        ));
    };
    if path.starts_with("sass:") {
        return Err(format!(
            "{file}: the built-in Sass module {path:?} is not supported (construct: @use {path})"
        ));
    }
    if rest.contains("with") && rest.contains('(') {
        return Err(format!(
            "{file}: `@use {path:?} with (...)` configuration is not supported \
             (construct: @use ... with)"
        ));
    }
    let namespace = if rest.is_empty() {
        UseNamespace::Default
    } else if let Some(alias) = rest.strip_prefix("as") {
        let alias = alias.trim();
        if alias == "*" {
            UseNamespace::Star
        } else if !alias.is_empty() && alias.bytes().all(is_ident_byte) {
            UseNamespace::Named(alias.to_string())
        } else {
            return Err(format!(
                "{file}: cannot parse `@use {path:?} as {alias}`"
            ));
        }
    } else {
        return Err(format!("{file}: cannot parse `@use {path:?} {rest}`"));
    };
    Ok(Item::Use { path, namespace })
}

fn parse_mixin_params(
    raw: &str,
    mixin: &str,
    file: &str,
) -> Result<Vec<(String, Option<String>)>, String> {
    let mut params = Vec::new();
    for part in split_top_level_commas(raw) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let Some(rest) = part.strip_prefix('$') else {
            return Err(format!(
                "{file}: cannot parse parameter {part:?} of `@mixin {mixin}`; expected `$name` \
                 or `$name: default`"
            ));
        };
        if rest.ends_with("...") {
            return Err(format!(
                "{file}: variadic parameters (`{part}`) of `@mixin {mixin}` are not supported \
                 (construct: variadic arguments)"
            ));
        }
        match rest.split_once(':') {
            Some((name, default)) => {
                params.push((name.trim().to_string(), Some(default.trim().to_string())));
            }
            None => params.push((rest.trim().to_string(), None)),
        }
    }
    Ok(params)
}

fn parse_rule_or_declaration(
    text: &str,
    bytes: &[u8],
    pos: &mut usize,
    file: &str,
) -> Result<Item, String> {
    let start = *pos;
    let (end, found) = scan_item_end(bytes, start, file)?;
    match found {
        b'{' => {
            let selector = text[start..end].trim().to_string();
            check_no_interpolation(&selector, file)?;
            for part in split_top_level_commas(&selector) {
                let part = part.trim();
                if let Some(rest) = part.strip_prefix('%')
                    && rest
                        .as_bytes()
                        .first()
                        .is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'-')
                {
                    return Err(format!(
                        "{file}: placeholder selector {part:?} is not supported \
                         (construct: %placeholder)"
                    ));
                }
            }
            *pos = end + 1;
            let body = parse_block(text, bytes, pos, file, false)?;
            Ok(Item::Rule { selector, body })
        }
        b';' | b'}' | 0 => {
            let declaration = text[start..end].trim();
            *pos = if found == b';' { end + 1 } else { end };
            let Some(colon) = find_colon(declaration) else {
                return Err(format!(
                    "{file}: cannot parse {declaration:?} as a declaration"
                ));
            };
            let property = declaration[..colon].trim().to_string();
            let value = declaration[colon + 1..].trim().to_string();
            check_no_interpolation(declaration, file)?;
            if property.is_empty() {
                return Err(format!(
                    "{file}: declaration {declaration:?} has no property name"
                ));
            }
            Ok(Item::Declaration { property, value })
        }
        _ => unreachable!("scan_item_end returns only {{ ; }} or EOF"),
    }
}

/// The first `:` outside strings/parens/brackets.
fn find_colon(declaration: &str) -> Option<usize> {
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
            b'[' => {
                index = match find_matching(bytes, index, b'[', b']', "declaration") {
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

/// Splits on top-level commas (string/paren/bracket aware).
fn split_top_level_commas(value: &str) -> Vec<&str> {
    let bytes = value.as_bytes();
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'"' | b'\'' => index = skip_string(bytes, index),
            b'(' => {
                index = match find_matching(bytes, index, b'(', b')', "value") {
                    Ok(close) => close + 1,
                    Err(_) => bytes.len(),
                }
            }
            b'[' => {
                index = match find_matching(bytes, index, b'[', b']', "value") {
                    Ok(close) => close + 1,
                    Err(_) => bytes.len(),
                }
            }
            b',' => {
                parts.push(&value[start..index]);
                index += 1;
                start = index;
            }
            _ => index += 1,
        }
    }
    parts.push(&value[start..]);
    parts
}

// ---------------------------------------------------------------------------
// Evaluation: item tree -> output nodes.
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum OutNode {
    Rule {
        selector: String,
        decls: Vec<String>,
    },
    AtBlock {
        header: String,
        children: Vec<OutNode>,
    },
    AtDecls {
        header: String,
        decls: Vec<String>,
    },
    Raw(String),
}

struct MixinDef {
    params: Vec<(String, Option<String>)>,
    body: Rc<Vec<Item>>,
    /// `None` = defined in the root file (evaluated against the current root
    /// scope); `Some(path)` = defined in a `@use`d module.
    owner: Option<PathBuf>,
}

#[derive(Default)]
struct Module {
    vars: HashMap<String, String>,
    mixins: HashMap<String, Rc<MixinDef>>,
    namespaces: HashMap<String, PathBuf>,
}

struct FileCtx {
    file: PathBuf,
    /// Scope stack; `[0]` is the file-global scope.
    scopes: Vec<HashMap<String, String>>,
    mixins: HashMap<String, Rc<MixinDef>>,
    namespaces: HashMap<String, PathBuf>,
}

impl FileCtx {
    fn new(file: PathBuf) -> Self {
        Self {
            file,
            scopes: vec![HashMap::new()],
            mixins: HashMap::new(),
            namespaces: HashMap::new(),
        }
    }

    fn lookup(&self, name: &str) -> Option<&str> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).map(String::as_str))
    }

    fn declare(&mut self, name: String, value: String) {
        // Sass assignment semantics: an existing variable in an outer scope is
        // reassigned; otherwise the current scope declares it.
        for scope in self.scopes.iter_mut().rev() {
            if let Some(existing) = scope.get_mut(&name) {
                *existing = value;
                return;
            }
        }
        self.scopes
            .last_mut()
            .expect("scope stack is never empty")
            .insert(name, value);
    }

    fn display(&self) -> String {
        self.file.display().to_string()
    }
}

struct Compiler<'o> {
    options: &'o ScssOptions,
    root_file: PathBuf,
    modules: HashMap<PathBuf, Module>,
    loading: Vec<PathBuf>,
    loaded_files: Vec<PathBuf>,
}

impl Compiler<'_> {
    fn eval_top_level(
        &mut self,
        items: &[Item],
        ctx: &mut FileCtx,
        out: &mut Vec<OutNode>,
    ) -> Result<(), String> {
        for item in items {
            match item {
                Item::VarDecl {
                    name,
                    value,
                    default,
                } => self.eval_var_decl(name, value, *default, ctx)?,
                Item::Declaration { property, .. } => {
                    return Err(format!(
                        "{}: declaration `{property}: ...` outside a rule",
                        ctx.display()
                    ));
                }
                Item::Rule { selector, body } => {
                    self.eval_rule(selector, body, ctx, &[], out)?;
                }
                Item::AtBlock {
                    name,
                    prelude,
                    body,
                } => self.eval_at_block(name, prelude, body, ctx, &[], out)?,
                Item::Use { path, namespace } => self.eval_use(path, namespace, ctx, out)?,
                Item::Import { target } => self.eval_import(target, ctx, out)?,
                Item::RawImport { statement } => out.push(OutNode::Raw(statement.clone())),
                Item::MixinDecl { name, params, body } => {
                    self.declare_mixin(name, params, body, ctx);
                }
                Item::Include { target, args } => {
                    let (definition, mut mixin_ctx) = self.prepare_include(target, args, ctx)?;
                    self.eval_top_level(&definition.body.clone(), &mut mixin_ctx, out)?;
                }
                Item::RawStatement { statement } => out.push(OutNode::Raw(statement.clone())),
            }
        }
        Ok(())
    }

    fn eval_var_decl(
        &mut self,
        name: &str,
        value: &str,
        default: bool,
        ctx: &mut FileCtx,
    ) -> Result<(), String> {
        if default && ctx.lookup(name).is_some() {
            return Ok(());
        }
        let evaluated = self.eval_value(value, ctx)?;
        ctx.declare(name.to_string(), evaluated);
        Ok(())
    }

    fn declare_mixin(
        &mut self,
        name: &str,
        params: &[(String, Option<String>)],
        body: &Rc<Vec<Item>>,
        ctx: &mut FileCtx,
    ) {
        let owner = if ctx.file == self.root_file {
            None
        } else {
            Some(ctx.file.clone())
        };
        ctx.mixins.insert(
            name.to_string(),
            Rc::new(MixinDef {
                params: params.to_vec(),
                body: Rc::clone(body),
                owner,
            }),
        );
    }

    /// Resolves an `@include` target and builds the evaluation context for the
    /// mixin body (defining module's scope + bound arguments).
    fn prepare_include(
        &mut self,
        target: &str,
        args: &[String],
        ctx: &mut FileCtx,
    ) -> Result<(Rc<MixinDef>, FileCtx), String> {
        for argument in args {
            if argument.starts_with('$') && find_colon(argument).is_some() {
                return Err(format!(
                    "{}: keyword arguments in `@include {target}` are not supported \
                     (construct: keyword arguments)",
                    ctx.display()
                ));
            }
        }
        let definition = match target.split_once('.') {
            Some((namespace, member)) => {
                let Some(module_path) = ctx.namespaces.get(namespace) else {
                    return Err(format!(
                        "{}: `@include {target}` references unknown namespace `{namespace}`",
                        ctx.display()
                    ));
                };
                let module = self
                    .modules
                    .get(module_path)
                    .expect("registered namespace always has a module");
                module.mixins.get(member).cloned().ok_or_else(|| {
                    format!(
                        "{}: mixin `{member}` is not defined in `{namespace}`",
                        ctx.display()
                    )
                })?
            }
            None => ctx.mixins.get(target).cloned().ok_or_else(|| {
                format!("{}: mixin `{target}` is not defined", ctx.display())
            })?,
        };
        // Evaluate arguments in the CALLER's scope before switching contexts.
        let mut evaluated_args = Vec::new();
        for argument in args {
            evaluated_args.push(self.eval_value(argument, ctx)?);
        }
        if evaluated_args.len() > definition.params.len() {
            return Err(format!(
                "{}: `@include {target}` passes {} arguments but the mixin takes {}",
                ctx.display(),
                evaluated_args.len(),
                definition.params.len()
            ));
        }
        let mut mixin_ctx = match &definition.owner {
            // A mixin can be included by its own module's top-level CSS while
            // that module is still being evaluated (before registration); the
            // live context IS the owner scope then.
            Some(module_path) if *module_path != ctx.file => {
                let module = self.modules.get(module_path).ok_or_else(|| {
                    format!(
                        "{}: mixin `{target}` belongs to a module still being loaded ({})",
                        ctx.display(),
                        module_path.display()
                    )
                })?;
                let mut owner_ctx = FileCtx::new(module_path.clone());
                owner_ctx.scopes[0] = module.vars.clone();
                owner_ctx.mixins = module.mixins.clone();
                owner_ctx.namespaces = module.namespaces.clone();
                owner_ctx
            }
            _ => {
                let mut owner_ctx = FileCtx::new(ctx.file.clone());
                owner_ctx.scopes[0] = ctx.scopes[0].clone();
                owner_ctx.mixins = ctx.mixins.clone();
                owner_ctx.namespaces = ctx.namespaces.clone();
                owner_ctx
            }
        };
        let mut bindings = HashMap::new();
        for (index, (parameter, default)) in definition.params.iter().enumerate() {
            match evaluated_args.get(index) {
                Some(value) => {
                    bindings.insert(parameter.clone(), value.clone());
                }
                None => match default {
                    Some(default) => {
                        // Defaults are evaluated in the mixin's own scope.
                        let value = self.eval_value(default, &mixin_ctx)?;
                        bindings.insert(parameter.clone(), value);
                    }
                    None => {
                        return Err(format!(
                            "{}: `@include {target}` is missing the argument `${parameter}`",
                            ctx.display()
                        ));
                    }
                },
            }
        }
        mixin_ctx.scopes.push(bindings);
        Ok((definition, mixin_ctx))
    }

    fn eval_rule(
        &mut self,
        selector: &str,
        body: &[Item],
        ctx: &mut FileCtx,
        parent: &[String],
        out: &mut Vec<OutNode>,
    ) -> Result<(), String> {
        let resolved = resolve_selectors(selector, parent, &ctx.display())?;
        let rule_index = out.len();
        out.push(OutNode::Rule {
            selector: resolved.join(", "),
            decls: Vec::new(),
        });
        ctx.scopes.push(HashMap::new());
        let result = self.eval_rule_body(body, ctx, &resolved, out, rule_index);
        ctx.scopes.pop();
        result
    }

    /// Evaluates a rule body: declarations sink into `out[rule_index]`, nested
    /// rules/at-blocks append after it, includes splice the mixin body here.
    fn eval_rule_body(
        &mut self,
        items: &[Item],
        ctx: &mut FileCtx,
        selector: &[String],
        out: &mut Vec<OutNode>,
        rule_index: usize,
    ) -> Result<(), String> {
        for item in items {
            match item {
                Item::VarDecl {
                    name,
                    value,
                    default,
                } => self.eval_var_decl(name, value, *default, ctx)?,
                Item::Declaration { property, value } => {
                    let rendered = self.eval_declaration(property, value, ctx)?;
                    let OutNode::Rule { decls, .. } = &mut out[rule_index] else {
                        unreachable!("rule_index always points at a Rule node");
                    };
                    decls.push(rendered);
                }
                Item::Rule {
                    selector: nested,
                    body,
                } => self.eval_rule(nested, body, ctx, selector, out)?,
                Item::AtBlock {
                    name,
                    prelude,
                    body,
                } => self.eval_at_block(name, prelude, body, ctx, selector, out)?,
                Item::MixinDecl { name, params, body } => {
                    self.declare_mixin(name, params, body, ctx);
                }
                Item::Include { target, args } => {
                    let (definition, mut mixin_ctx) = self.prepare_include(target, args, ctx)?;
                    self.eval_rule_body(
                        &definition.body.clone(),
                        &mut mixin_ctx,
                        selector,
                        out,
                        rule_index,
                    )?;
                }
                Item::Use { .. } => {
                    return Err(format!(
                        "{}: `@use` is only allowed at the top level",
                        ctx.display()
                    ));
                }
                Item::Import { target } | Item::RawImport {
                    statement: target, ..
                } => {
                    return Err(format!(
                        "{}: `@import {target:?}` inside a rule is not supported",
                        ctx.display()
                    ));
                }
                Item::RawStatement { statement } => {
                    return Err(format!(
                        "{}: {statement:?} inside a rule is not supported",
                        ctx.display()
                    ));
                }
            }
        }
        Ok(())
    }

    fn eval_at_block(
        &mut self,
        name: &str,
        prelude: &str,
        body: &[Item],
        ctx: &mut FileCtx,
        parent: &[String],
        out: &mut Vec<OutNode>,
    ) -> Result<(), String> {
        match name {
            "media" | "supports" | "container" | "layer" => {
                let prelude = self.eval_prelude(prelude, ctx)?;
                let header = if prelude.is_empty() {
                    format!("@{name}")
                } else {
                    format!("@{name} {prelude}")
                };
                let mut children = Vec::new();
                self.eval_conditional_body(body, ctx, parent, &mut children)?;
                prune_empty(&mut children);
                if !children.is_empty() {
                    out.push(OutNode::AtBlock { header, children });
                }
                Ok(())
            }
            "font-face" | "page" => {
                let header = if prelude.is_empty() {
                    format!("@{name}")
                } else {
                    format!("@{name} {prelude}")
                };
                let mut decls = Vec::new();
                for item in body {
                    match item {
                        Item::Declaration { property, value } => {
                            decls.push(self.eval_declaration(property, value, ctx)?);
                        }
                        Item::VarDecl {
                            name: variable,
                            value,
                            default,
                        } => self.eval_var_decl(variable, value, *default, ctx)?,
                        _ => {
                            return Err(format!(
                                "{}: only declarations are supported inside `@{name}`",
                                ctx.display()
                            ));
                        }
                    }
                }
                out.push(OutNode::AtDecls { header, decls });
                Ok(())
            }
            _ if name.eq_ignore_ascii_case("keyframes")
                || name.to_ascii_lowercase().ends_with("-keyframes") =>
            {
                let animation = prelude.trim();
                if animation.is_empty() {
                    return Err(format!("{}: `@{name}` without a name", ctx.display()));
                }
                let mut frames = Vec::new();
                for item in body {
                    match item {
                        Item::Rule { selector, body } => {
                            let frame = normalize_frame_selector(selector);
                            let frame_index = frames.len();
                            frames.push(OutNode::Rule {
                                selector: frame,
                                decls: Vec::new(),
                            });
                            ctx.scopes.push(HashMap::new());
                            let result = self.eval_keyframe_body(
                                body,
                                ctx,
                                &mut frames,
                                frame_index,
                            );
                            ctx.scopes.pop();
                            result?;
                        }
                        Item::VarDecl {
                            name: variable,
                            value,
                            default,
                        } => self.eval_var_decl(variable, value, *default, ctx)?,
                        _ => {
                            return Err(format!(
                                "{}: only `<selector> {{ ... }}` frames are supported inside \
                                 `@{name}`",
                                ctx.display()
                            ));
                        }
                    }
                }
                out.push(OutNode::AtBlock {
                    header: format!("@{name} {animation}"),
                    children: frames,
                });
                Ok(())
            }
            _ => Err(format!(
                "{}: unsupported at-rule `@{name}` (construct: @{name})",
                ctx.display()
            )),
        }
    }

    /// A keyframe frame body: declarations only (plus local variables).
    fn eval_keyframe_body(
        &mut self,
        items: &[Item],
        ctx: &mut FileCtx,
        frames: &mut [OutNode],
        frame_index: usize,
    ) -> Result<(), String> {
        for item in items {
            match item {
                Item::Declaration { property, value } => {
                    let rendered = self.eval_declaration(property, value, ctx)?;
                    let OutNode::Rule { decls, .. } = &mut frames[frame_index] else {
                        unreachable!("frame_index always points at a Rule node");
                    };
                    decls.push(rendered);
                }
                Item::VarDecl {
                    name,
                    value,
                    default,
                } => self.eval_var_decl(name, value, *default, ctx)?,
                _ => {
                    return Err(format!(
                        "{}: only declarations are supported inside a keyframe frame",
                        ctx.display()
                    ));
                }
            }
        }
        Ok(())
    }

    /// The body of a conditional group (`@media` and friends): nested rules
    /// evaluate normally; bare declarations attach to the enclosing selector
    /// (`#x {{ @media ... {{ padding: 0; }} }}`).
    fn eval_conditional_body(
        &mut self,
        items: &[Item],
        ctx: &mut FileCtx,
        parent: &[String],
        out: &mut Vec<OutNode>,
    ) -> Result<(), String> {
        let has_declarations = items
            .iter()
            .any(|item| matches!(item, Item::Declaration { .. } | Item::Include { .. }));
        if has_declarations && !parent.is_empty() {
            // Direct declarations re-nest under the parent selector.
            let rule_index = out.len();
            out.push(OutNode::Rule {
                selector: parent.join(", "),
                decls: Vec::new(),
            });
            ctx.scopes.push(HashMap::new());
            let result = self.eval_rule_body(items, ctx, parent, out, rule_index);
            ctx.scopes.pop();
            return result;
        }
        for item in items {
            match item {
                Item::VarDecl {
                    name,
                    value,
                    default,
                } => self.eval_var_decl(name, value, *default, ctx)?,
                Item::Declaration { property, .. } => {
                    return Err(format!(
                        "{}: declaration `{property}: ...` directly inside a conditional \
                         at-rule with no enclosing selector",
                        ctx.display()
                    ));
                }
                Item::Rule { selector, body } => {
                    self.eval_rule(selector, body, ctx, parent, out)?;
                }
                Item::AtBlock {
                    name,
                    prelude,
                    body,
                } => self.eval_at_block(name, prelude, body, ctx, parent, out)?,
                Item::MixinDecl { name, params, body } => {
                    self.declare_mixin(name, params, body, ctx);
                }
                Item::Include { target, args } => {
                    let (definition, mut mixin_ctx) = self.prepare_include(target, args, ctx)?;
                    self.eval_conditional_body(
                        &definition.body.clone(),
                        &mut mixin_ctx,
                        parent,
                        out,
                    )?;
                }
                Item::Use { .. } => {
                    return Err(format!(
                        "{}: `@use` is only allowed at the top level",
                        ctx.display()
                    ));
                }
                Item::Import { .. } | Item::RawImport { .. } | Item::RawStatement { .. } => {
                    return Err(format!(
                        "{}: `@import` inside a conditional at-rule is not supported",
                        ctx.display()
                    ));
                }
            }
        }
        Ok(())
    }

    fn eval_declaration(
        &self,
        property: &str,
        value: &str,
        ctx: &FileCtx,
    ) -> Result<String, String> {
        // Custom properties keep their value verbatim (Sass treats them as
        // opaque except for interpolation, which is rejected at parse).
        if property.starts_with("--") {
            return Ok(format!("{property}: {value}"));
        }
        let evaluated = self.eval_value(value, ctx)?;
        Ok(format!("{property}: {evaluated}"))
    }

    /// Evaluates an at-rule prelude: each parenthesized `(feature: value)`
    /// group's value is a full Sass expression (`(min-width: $breakpoint-sm)`,
    /// `(min-width: 2 * 400px)`); everything else passes through with `$var`
    /// references substituted and whitespace collapsed.
    fn eval_prelude(&self, prelude: &str, ctx: &FileCtx) -> Result<String, String> {
        let bytes = prelude.as_bytes();
        let mut out = String::with_capacity(prelude.len());
        let mut index = 0usize;
        while index < bytes.len() {
            match bytes[index] {
                b'"' | b'\'' => {
                    let end = skip_string(bytes, index);
                    out.push_str(&prelude[index..end]);
                    index = end;
                }
                b'(' => {
                    let close =
                        find_matching(bytes, index, b'(', b')', &ctx.display())?;
                    let inner = &prelude[index + 1..close];
                    match find_colon(inner) {
                        Some(colon) => {
                            let feature = inner[..colon].trim();
                            let value = self.eval_value(&inner[colon + 1..], ctx)?;
                            out.push('(');
                            out.push_str(feature);
                            out.push_str(": ");
                            out.push_str(&value);
                            out.push(')');
                        }
                        None => {
                            out.push('(');
                            out.push_str(&self.eval_prelude(inner, ctx)?);
                            out.push(')');
                        }
                    }
                    index = close + 1;
                }
                _ => {
                    let start = index;
                    while index < bytes.len()
                        && !matches!(bytes[index], b'(' | b'"' | b'\'')
                    {
                        index += 1;
                    }
                    out.push_str(&self.substitute_prelude_vars(
                        &prelude[start..index],
                        ctx,
                        prelude,
                    )?);
                }
            }
        }
        Ok(collapse_whitespace(&out))
    }

    /// Substitutes `$var` / `ns.$var` references in a prelude fragment.
    fn substitute_prelude_vars(
        &self,
        fragment: &str,
        ctx: &FileCtx,
        prelude: &str,
    ) -> Result<String, String> {
        let bytes = fragment.as_bytes();
        let mut out = String::with_capacity(prelude.len());
        let mut index = 0usize;
        while index < bytes.len() {
            match bytes[index] {
                b'"' | b'\'' => {
                    let end = skip_string(bytes, index);
                    out.push_str(&fragment[index..end]);
                    index = end;
                }
                b'$' => {
                    let end = ident_end(bytes, index + 1);
                    if end == index + 1 {
                        out.push('$');
                        index += 1;
                        continue;
                    }
                    let name = &fragment[index + 1..end];
                    // A `ns.` immediately before this `$` was already copied;
                    // detect and rewrite it.
                    let namespace = out
                        .strip_suffix('.')
                        .and_then(|head| {
                            let ns_start = head
                                .rfind(|c: char| !is_ident_byte(c as u8))
                                .map_or(0, |position| position + 1);
                            let candidate = &head[ns_start..];
                            ctx.namespaces
                                .contains_key(candidate)
                                .then(|| (head[..ns_start].len(), candidate.to_string()))
                        });
                    let value = match namespace {
                        Some((keep, namespace)) => {
                            let value =
                                self.lookup_namespaced_var(&namespace, name, ctx)?;
                            out.truncate(keep);
                            value
                        }
                        None => ctx
                            .lookup(name)
                            .map(str::to_string)
                            .ok_or_else(|| {
                                format!(
                                    "{}: undefined variable `${name}` in at-rule prelude \
                                     {prelude:?}",
                                    ctx.display()
                                )
                            })?,
                    };
                    out.push_str(&value);
                    index = end;
                }
                byte if byte.is_ascii_whitespace() => {
                    if !out.ends_with(' ') && !out.is_empty() {
                        out.push(' ');
                    }
                    index += 1;
                }
                byte => {
                    out.push(byte as char);
                    index += 1;
                }
            }
        }
        Ok(out)
    }

    fn lookup_namespaced_var(
        &self,
        namespace: &str,
        name: &str,
        ctx: &FileCtx,
    ) -> Result<String, String> {
        let Some(module_path) = ctx.namespaces.get(namespace) else {
            return Err(format!(
                "{}: unknown namespace `{namespace}` (in `{namespace}.${name}`)",
                ctx.display()
            ));
        };
        let module = self
            .modules
            .get(module_path)
            .expect("registered namespace always has a module");
        module.vars.get(name).cloned().ok_or_else(|| {
            format!(
                "{}: variable `${name}` is not defined in `{namespace}`",
                ctx.display()
            )
        })
    }

    // -- @use / @import ----------------------------------------------------

    fn eval_use(
        &mut self,
        path: &str,
        namespace: &UseNamespace,
        ctx: &mut FileCtx,
        out: &mut Vec<OutNode>,
    ) -> Result<(), String> {
        let resolved = self.resolve_scss_target(path, &ctx.file, &ctx.display())?;
        self.load_module(&resolved, out)?;
        match namespace {
            UseNamespace::Named(alias) => {
                ctx.namespaces.insert(alias.clone(), resolved);
            }
            UseNamespace::Default => {
                let stem = resolved
                    .file_stem()
                    .and_then(|value| value.to_str())
                    .map(|value| value.trim_start_matches('_').to_string())
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| {
                        format!(
                            "{}: cannot derive a namespace from `@use {path:?}`; \
                             add `as <name>`",
                            ctx.display()
                        )
                    })?;
                ctx.namespaces.insert(stem, resolved);
            }
            UseNamespace::Star => {
                let module = self
                    .modules
                    .get(&resolved)
                    .expect("just-loaded module is registered");
                for (name, value) in &module.vars {
                    if ctx.lookup(name).is_none() {
                        ctx.scopes[0].insert(name.clone(), value.clone());
                    }
                }
                let mixins: Vec<_> = module
                    .mixins
                    .iter()
                    .map(|(name, definition)| (name.clone(), Rc::clone(definition)))
                    .collect();
                for (name, definition) in mixins {
                    ctx.mixins.entry(name).or_insert(definition);
                }
            }
        }
        Ok(())
    }

    /// Loads (at most once) and evaluates a `@use`d file; its CSS is emitted
    /// into `out` at the first use.
    fn load_module(&mut self, resolved: &PathBuf, out: &mut Vec<OutNode>) -> Result<(), String> {
        if self.modules.contains_key(resolved) {
            return Ok(());
        }
        if self.loading.contains(resolved) {
            return Err(format!(
                "module loop: {} is already being loaded",
                resolved.display()
            ));
        }
        let source = fs::read_to_string(resolved)
            .map_err(|error| format!("cannot read {}: {error}", resolved.display()))?;
        self.loaded_files.push(resolved.clone());
        let display = resolved.display().to_string();
        let clean = strip_sass_comments(&source);
        let items = parse_block_str(&clean, &display)?;
        self.loading.push(resolved.clone());
        let mut module_ctx = FileCtx::new(resolved.clone());
        let result = self.eval_top_level(&items, &mut module_ctx, out);
        self.loading.pop();
        result?;
        self.modules.insert(
            resolved.clone(),
            Module {
                vars: module_ctx.scopes.swap_remove(0),
                mixins: module_ctx.mixins,
                namespaces: module_ctx.namespaces,
            },
        );
        Ok(())
    }

    /// `@import './x'` of a Sass partial: evaluated IN PLACE in the importer's
    /// scope (variables and mixins become the importer's), CSS emitted here.
    fn eval_import(
        &mut self,
        target: &str,
        ctx: &mut FileCtx,
        out: &mut Vec<OutNode>,
    ) -> Result<(), String> {
        let resolved = self.resolve_scss_target(target, &ctx.file, &ctx.display())?;
        if self.loading.contains(&resolved) || resolved == self.root_file {
            return Err(format!(
                "import loop: {} is already being loaded",
                resolved.display()
            ));
        }
        let source = fs::read_to_string(&resolved)
            .map_err(|error| format!("cannot read {}: {error}", resolved.display()))?;
        self.loaded_files.push(resolved.clone());
        let display = resolved.display().to_string();
        let clean = strip_sass_comments(&source);
        let items = parse_block_str(&clean, &display)?;
        self.loading.push(resolved.clone());
        // The imported file's own relative urls must resolve against ITS
        // directory: swap the context's file while evaluating.
        let previous = std::mem::replace(&mut ctx.file, resolved);
        let result = self.eval_top_level(&items, ctx, out);
        ctx.file = previous;
        self.loading.pop();
        result
    }

    /// Resolves a `@use`/`@import` target to a real file: root-relative `/...`
    /// against the configured project root, everything else against the
    /// current file's directory, trying the `_partial` and index conventions.
    fn resolve_scss_target(
        &self,
        target: &str,
        from: &Path,
        display: &str,
    ) -> Result<PathBuf, String> {
        let base = if let Some(rest) = target.strip_prefix('/') {
            let Some(root) = &self.options.root else {
                return Err(format!(
                    "{display}: root-relative `@use {target:?}` needs a known project root \
                     (build without one cannot resolve it)"
                ));
            };
            root.join(rest)
        } else {
            let directory = from.parent().unwrap_or_else(|| Path::new("."));
            directory.join(target)
        };
        let mut candidates: Vec<PathBuf> = Vec::new();
        let extension = base.extension().and_then(|value| value.to_str());
        let push_with_partial = |candidates: &mut Vec<PathBuf>, path: PathBuf| {
            if let (Some(parent), Some(name)) =
                (path.parent(), path.file_name().and_then(|value| value.to_str()))
            {
                candidates.push(path.clone());
                candidates.push(parent.join(format!("_{name}")));
            } else {
                candidates.push(path);
            }
        };
        match extension {
            Some("scss") => push_with_partial(&mut candidates, base.clone()),
            Some("sass") => {
                return Err(format!(
                    "{display}: `@use {target:?}` targets the indented `.sass` syntax, \
                     which is not supported (construct: indented syntax)"
                ));
            }
            _ => {
                push_with_partial(&mut candidates, base.with_extension_appended("scss"));
                candidates.push(base.join("_index.scss"));
                candidates.push(base.join("index.scss"));
            }
        }
        for candidate in &candidates {
            if candidate.is_file() {
                return Ok(candidate
                    .canonicalize()
                    .unwrap_or_else(|_| candidate.clone()));
            }
        }
        // A `.sass` sibling exists: name the real problem, not a missing file.
        let sass_sibling = base.with_extension_appended("sass");
        if sass_sibling.is_file() {
            return Err(format!(
                "{display}: `@use {target:?}` resolves to {}, but the indented `.sass` \
                 syntax is not supported (construct: indented syntax)",
                sass_sibling.display()
            ));
        }
        Err(format!(
            "{display}: cannot resolve `@use {target:?}` (tried {})",
            candidates
                .iter()
                .map(|candidate| candidate.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ))
    }

    // -- Value evaluation ----------------------------------------------------

    fn eval_value(&self, value: &str, ctx: &FileCtx) -> Result<String, String> {
        let value = value.trim();
        if value.is_empty() {
            return Ok(String::new());
        }
        let tokens = tokenize_value(value, &ctx.display())?;
        let mut parser = ValueParser {
            compiler: self,
            ctx,
            tokens: &tokens,
            index: 0,
        };
        let result = parser.parse_comma_list()?;
        if parser.index < tokens.len() {
            return Err(format!(
                "{}: cannot parse the value {value:?}",
                ctx.display()
            ));
        }
        Ok(render_val(&result))
    }
}

/// Appends an extension without replacing an existing suffix (`a.b` ->
/// `a.b.scss`), unlike [`Path::with_extension`].
trait WithExtensionAppended {
    fn with_extension_appended(&self, extension: &str) -> PathBuf;
}

impl WithExtensionAppended for PathBuf {
    fn with_extension_appended(&self, extension: &str) -> PathBuf {
        let mut os = self.clone().into_os_string();
        os.push(".");
        os.push(extension);
        PathBuf::from(os)
    }
}

// ---------------------------------------------------------------------------
// Selector resolution (nesting and `&`).
// ---------------------------------------------------------------------------

fn resolve_selectors(
    selector: &str,
    parent: &[String],
    file: &str,
) -> Result<Vec<String>, String> {
    let children: Vec<String> = split_top_level_commas(selector)
        .into_iter()
        .map(collapse_whitespace)
        .filter(|part| !part.is_empty())
        .collect();
    if children.is_empty() {
        return Err(format!("{file}: empty selector"));
    }
    let mut resolved = Vec::new();
    if parent.is_empty() {
        for child in &children {
            if selector_has_ampersand(child) {
                return Err(format!(
                    "{file}: `&` in the top-level selector {child:?} has no parent"
                ));
            }
            resolved.push(child.clone());
        }
        return Ok(resolved);
    }
    // Parent-major order, matching dart-sass:
    // `.a, .b { &:hover, .x & }` -> `.a:hover, .x .a, .b:hover, .x .b`.
    for parent_selector in parent {
        for child in &children {
            if selector_has_ampersand(child) {
                resolved.push(replace_ampersand(child, parent_selector));
            } else {
                resolved.push(format!("{parent_selector} {child}"));
            }
        }
    }
    Ok(resolved)
}

fn collapse_whitespace(part: &str) -> String {
    part.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn selector_has_ampersand(selector: &str) -> bool {
    let bytes = selector.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'"' | b'\'' => index = skip_string(bytes, index),
            b'[' => {
                index = match find_matching(bytes, index, b'[', b']', "selector") {
                    Ok(close) => close + 1,
                    Err(_) => bytes.len(),
                }
            }
            b'&' => return true,
            _ => index += 1,
        }
    }
    false
}

fn replace_ampersand(selector: &str, parent: &str) -> String {
    let bytes = selector.as_bytes();
    let mut out = String::with_capacity(selector.len() + parent.len());
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'"' | b'\'' => {
                let end = skip_string(bytes, index);
                out.push_str(&selector[index..end]);
                index = end;
            }
            b'[' => {
                let end = match find_matching(bytes, index, b'[', b']', "selector") {
                    Ok(close) => close + 1,
                    Err(_) => bytes.len(),
                };
                out.push_str(&selector[index..end]);
                index = end;
            }
            b'&' => {
                out.push_str(parent);
                index += 1;
            }
            byte => {
                out.push(byte as char);
                index += 1;
            }
        }
    }
    out
}

fn normalize_frame_selector(selector: &str) -> String {
    split_top_level_commas(selector)
        .into_iter()
        .map(collapse_whitespace)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(", ")
}

// ---------------------------------------------------------------------------
// Output pruning and rendering.
// ---------------------------------------------------------------------------

fn prune_empty(nodes: &mut Vec<OutNode>) {
    for node in nodes.iter_mut() {
        if let OutNode::AtBlock { children, .. } = node {
            prune_empty(children);
        }
    }
    nodes.retain(|node| match node {
        OutNode::Rule { decls, .. } => !decls.is_empty(),
        OutNode::AtBlock { header, children } => {
            // Keyframes stay even when a frame list is empty (they never are
            // in practice); conditional groups without content are dropped.
            !children.is_empty() || header.contains("keyframes")
        }
        OutNode::AtDecls { decls, .. } => !decls.is_empty(),
        OutNode::Raw(_) => true,
    });
}

/// Renders nodes; empty rules/declaration blocks are skipped (dart-sass
/// omits them).
fn render_nodes(nodes: &[OutNode], indent: usize, out: &mut String) {
    for node in nodes {
        render_node(node, indent, out);
    }
}

fn render_node(node: &OutNode, indent: usize, out: &mut String) {
    let pad = "  ".repeat(indent);
    match node {
        OutNode::Rule { selector, decls } => {
            if decls.is_empty() {
                return;
            }
            out.push_str(&pad);
            out.push_str(selector);
            out.push_str(" {\n");
            for decl in decls {
                out.push_str(&pad);
                out.push_str("  ");
                out.push_str(decl);
                out.push_str(";\n");
            }
            out.push_str(&pad);
            out.push_str("}\n");
        }
        OutNode::AtBlock { header, children } => {
            out.push_str(&pad);
            out.push_str(header);
            out.push_str(" {\n");
            render_nodes(children, indent + 1, out);
            out.push_str(&pad);
            out.push_str("}\n");
        }
        OutNode::AtDecls { header, decls } => {
            if decls.is_empty() {
                return;
            }
            out.push_str(&pad);
            out.push_str(header);
            out.push_str(" {\n");
            for decl in decls {
                out.push_str(&pad);
                out.push_str("  ");
                out.push_str(decl);
                out.push_str(";\n");
            }
            out.push_str(&pad);
            out.push_str("}\n");
        }
        OutNode::Raw(statement) => {
            out.push_str(&pad);
            out.push_str(statement);
            out.push('\n');
        }
    }
}

// ---------------------------------------------------------------------------
// Value expressions.
// ---------------------------------------------------------------------------

/// Sass-only functions that would produce silently-wrong CSS if passed
/// through. Each is a hard error naming the construct.
const SASS_ONLY_FUNCTIONS: &[&str] = &[
    "darken",
    "lighten",
    "adjust-hue",
    "adjust-color",
    "scale-color",
    "change-color",
    "opacify",
    "fade-in",
    "transparentize",
    "fade-out",
    "complement",
    "mix",
    "map-get",
    "map-merge",
    "map-keys",
    "map-values",
    "nth",
    "join",
    "append",
    "zip",
    "index",
    "unquote",
    "quote",
    "str-length",
    "str-insert",
    "str-index",
    "str-slice",
    "to-upper-case",
    "to-lower-case",
    "percentage",
    "unit",
    "unitless",
    "comparable",
    "random",
    "if",
    "mixin-exists",
    "variable-exists",
    "function-exists",
    "type-of",
    "inspect",
    "length",
    "list-separator",
    "set-nth",
    "keywords",
    "call",
    "get-function",
    "selector-append",
    "selector-nest",
    "selector-replace",
];

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num { value: f64, unit: String },
    Ident(String),
    /// `name` + raw argument text (parens stripped).
    Call { name: String, args: String },
    /// `url(...)` with raw inner text.
    Url(String),
    /// A quoted string, verbatim including quotes.
    Str(String),
    Var { namespace: Option<String>, name: String },
    /// `#hex` (or any `#...` token).
    Hash(String),
    /// A parenthesized group's raw inner text.
    Paren(String),
    /// `+ - * /` — `spaced_before`/`spaced_after` drive list-vs-operator
    /// disambiguation for `+`/`-`.
    Op { op: char, spaced_after: bool },
    Bang(String),
    Comma,
    /// A unary minus applied to the next primary (`-$x`, `-(1 + 2)`).
    UnaryMinus,
}

/// One token plus whether whitespace preceded it.
#[derive(Debug, Clone)]
struct SpacedTok {
    spaced_before: bool,
    tok: Tok,
}

fn tokenize_value(value: &str, file: &str) -> Result<Vec<SpacedTok>, String> {
    let bytes = value.as_bytes();
    let mut tokens: Vec<SpacedTok> = Vec::new();
    let mut index = 0usize;
    let mut spaced = false;
    while index < bytes.len() {
        let byte = bytes[index];
        if byte.is_ascii_whitespace() {
            spaced = true;
            index += 1;
            continue;
        }
        let spaced_before = std::mem::take(&mut spaced);
        match byte {
            b'"' | b'\'' => {
                let end = skip_string(bytes, index);
                tokens.push(SpacedTok {
                    spaced_before,
                    tok: Tok::Str(value[index..end].to_string()),
                });
                index = end;
            }
            b',' => {
                tokens.push(SpacedTok {
                    spaced_before,
                    tok: Tok::Comma,
                });
                index += 1;
            }
            b'#' => {
                if bytes.get(index + 1) == Some(&b'{') {
                    return Err(format!(
                        "{file}: interpolation `#{{...}}` is not supported (in {value:?})"
                    ));
                }
                let end = ident_end(bytes, index + 1);
                tokens.push(SpacedTok {
                    spaced_before,
                    tok: Tok::Hash(value[index..end].to_string()),
                });
                index = end.max(index + 1);
            }
            b'$' => {
                let end = ident_end(bytes, index + 1);
                if end == index + 1 {
                    return Err(format!(
                        "{file}: `$` without a variable name in {value:?}"
                    ));
                }
                tokens.push(SpacedTok {
                    spaced_before,
                    tok: Tok::Var {
                        namespace: None,
                        name: value[index + 1..end].to_string(),
                    },
                });
                index = end;
            }
            b'(' => {
                let close = find_matching(bytes, index, b'(', b')', file)?;
                tokens.push(SpacedTok {
                    spaced_before,
                    tok: Tok::Paren(value[index + 1..close].to_string()),
                });
                index = close + 1;
            }
            b'!' => {
                let end = ident_end(bytes, index + 1);
                tokens.push(SpacedTok {
                    spaced_before,
                    tok: Tok::Bang(value[index..end].to_string()),
                });
                index = end.max(index + 1);
            }
            b'0'..=b'9' => {
                let (token, end) = scan_number(value, bytes, index, false)?;
                tokens.push(SpacedTok {
                    spaced_before,
                    tok: token,
                });
                index = end;
            }
            b'.' if bytes.get(index + 1).is_some_and(u8::is_ascii_digit) => {
                let (token, end) = scan_number(value, bytes, index, false)?;
                tokens.push(SpacedTok {
                    spaced_before,
                    tok: token,
                });
                index = end;
            }
            b'+' | b'-' | b'*' | b'/' | b'%' => {
                let next = bytes.get(index + 1).copied();
                let next_spaced = next.is_none_or(|byte| byte.is_ascii_whitespace());
                if byte == b'-' || byte == b'+' {
                    let follows_operand = next.is_some_and(|next_byte| {
                        next_byte.is_ascii_digit() || next_byte == b'.'
                    });
                    let prev_is_operand = matches!(
                        tokens.last().map(|token| &token.tok),
                        Some(
                            Tok::Num { .. }
                                | Tok::Ident(_)
                                | Tok::Call { .. }
                                | Tok::Paren(_)
                                | Tok::Var { .. }
                                | Tok::Url(_)
                                | Tok::Hash(_)
                                | Tok::Str(_)
                        )
                    );
                    if follows_operand && (!prev_is_operand || spaced_before) {
                        // A signed number literal (`-5vh`, `-45deg`, `+3`).
                        let (token, end) = scan_number(value, bytes, index, true)?;
                        tokens.push(SpacedTok {
                            spaced_before,
                            tok: token,
                        });
                        index = end;
                        continue;
                    }
                    if byte == b'-'
                        && next.is_some_and(|next_byte| {
                            next_byte.is_ascii_alphabetic()
                                || next_byte == b'-'
                                || next_byte >= 0x80
                        })
                        && (!prev_is_operand || spaced_before)
                    {
                        // `-webkit-box`, `--custom` — an identifier.
                        let end = ident_end(bytes, index);
                        let (token, consumed) =
                            scan_ident_token(value, bytes, index, end, file)?;
                        tokens.push(SpacedTok {
                            spaced_before,
                            tok: token,
                        });
                        index = consumed;
                        continue;
                    }
                    if byte == b'-'
                        && !next_spaced
                        && matches!(next, Some(b'$') | Some(b'('))
                        && (!prev_is_operand || spaced_before)
                    {
                        tokens.push(SpacedTok {
                            spaced_before,
                            tok: Tok::UnaryMinus,
                        });
                        index += 1;
                        continue;
                    }
                }
                tokens.push(SpacedTok {
                    spaced_before,
                    tok: Tok::Op {
                        op: byte as char,
                        spaced_after: next_spaced,
                    },
                });
                index += 1;
            }
            _ if byte.is_ascii_alphabetic() || byte == b'_' || byte >= 0x80 => {
                let end = ident_end(bytes, index);
                let (token, consumed) = scan_ident_token(value, bytes, index, end, file)?;
                tokens.push(SpacedTok {
                    spaced_before,
                    tok: token,
                });
                index = consumed;
            }
            _ => {
                return Err(format!(
                    "{file}: unexpected `{}` in value {value:?}",
                    byte as char
                ));
            }
        }
    }
    Ok(tokens)
}

/// Scans a number (optionally signed) with its unit.
fn scan_number(
    value: &str,
    bytes: &[u8],
    start: usize,
    signed: bool,
) -> Result<(Tok, usize), String> {
    let mut cursor = start;
    if signed {
        cursor += 1; // the sign byte
    }
    while cursor < bytes.len() && (bytes[cursor].is_ascii_digit() || bytes[cursor] == b'.') {
        cursor += 1;
    }
    let number: f64 = value[start..cursor]
        .parse()
        .map_err(|_| format!("cannot parse number {:?}", &value[start..cursor]))?;
    // Unit: an identifier or `%`.
    let unit_end = if bytes.get(cursor) == Some(&b'%') {
        cursor + 1
    } else {
        ident_end(bytes, cursor)
    };
    let unit = value[cursor..unit_end].to_string();
    Ok((
        Tok::Num {
            value: number,
            unit,
        },
        unit_end,
    ))
}

/// Scans an identifier and classifies it: plain ident, function call,
/// `url(...)`, namespaced variable (`ns.$x`), or a namespaced function call
/// (hard error).
fn scan_ident_token(
    value: &str,
    bytes: &[u8],
    start: usize,
    end: usize,
    file: &str,
) -> Result<(Tok, usize), String> {
    let name = &value[start..end];
    match bytes.get(end) {
        Some(b'(') => {
            let close = find_matching(bytes, end, b'(', b')', file)?;
            if name.eq_ignore_ascii_case("url") {
                return Ok((Tok::Url(value[end + 1..close].to_string()), close + 1));
            }
            Ok((
                Tok::Call {
                    name: name.to_string(),
                    args: value[end + 1..close].to_string(),
                },
                close + 1,
            ))
        }
        Some(b'.') => {
            let after = end + 1;
            if bytes.get(after) == Some(&b'$') {
                let var_end = ident_end(bytes, after + 1);
                if var_end == after + 1 {
                    return Err(format!(
                        "{file}: `{name}.$` without a variable name in {value:?}"
                    ));
                }
                return Ok((
                    Tok::Var {
                        namespace: Some(name.to_string()),
                        name: value[after + 1..var_end].to_string(),
                    },
                    var_end,
                ));
            }
            let member_end = ident_end(bytes, after);
            if member_end > after && bytes.get(member_end) == Some(&b'(') {
                let member = &value[after..member_end];
                return Err(format!(
                    "{file}: the namespaced function `{name}.{member}(...)` is not supported \
                     (construct: {name}.{member})"
                ));
            }
            // Not a module member — a plain dotted token (rare; keep as-is).
            Ok((Tok::Ident(name.to_string()), end))
        }
        _ => Ok((Tok::Ident(name.to_string()), end)),
    }
}

/// An evaluated value.
#[derive(Debug, Clone, PartialEq)]
enum Val {
    Num { value: f64, unit: String },
    Lit(String),
    /// `sep` is `','` or `' '`; `'/'` items render glued.
    List { items: Vec<Val>, sep: char },
}

fn render_val(value: &Val) -> String {
    match value {
        Val::Num { value, unit } => format!("{}{unit}", format_number(*value)),
        Val::Lit(text) => text.clone(),
        Val::List { items, sep } => {
            let rendered: Vec<String> = items.iter().map(render_val).collect();
            match sep {
                ',' => rendered.join(", "),
                _ => rendered.join(" "),
            }
        }
    }
}

/// dart-sass prints numbers rounded to 10 decimal places, trailing zeros
/// trimmed.
fn format_number(value: f64) -> String {
    let rounded = (value * 1e10).round() / 1e10;
    let rounded = if rounded == 0.0 { 0.0 } else { rounded };
    if rounded.fract() == 0.0 && rounded.abs() < 1e15 {
        format!("{}", rounded as i64)
    } else {
        let text = format!("{rounded:.10}");
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

struct ValueParser<'a> {
    compiler: &'a Compiler<'a>,
    ctx: &'a FileCtx,
    tokens: &'a [SpacedTok],
    index: usize,
}

impl ValueParser<'_> {
    fn file(&self) -> String {
        self.ctx.display()
    }

    fn peek(&self) -> Option<&SpacedTok> {
        self.tokens.get(self.index)
    }

    fn parse_comma_list(&mut self) -> Result<Val, String> {
        let mut items = vec![self.parse_space_list()?];
        while matches!(self.peek().map(|token| &token.tok), Some(Tok::Comma)) {
            self.index += 1;
            items.push(self.parse_space_list()?);
        }
        if items.len() == 1 {
            Ok(items.pop().expect("one item"))
        } else {
            Ok(Val::List { items, sep: ',' })
        }
    }

    fn parse_space_list(&mut self) -> Result<Val, String> {
        let mut items = Vec::new();
        loop {
            match self.peek().map(|token| &token.tok) {
                None | Some(Tok::Comma) => break,
                Some(Tok::Op { op: '/', .. }) => {
                    // A literal slash separator (`font: 12px/1.5`); glue it.
                    self.index += 1;
                    let Some(previous) = items.pop() else {
                        return Err(format!("{}: a value cannot start with `/`", self.file()));
                    };
                    let next = self.parse_sum()?;
                    items.push(Val::Lit(format!(
                        "{}/{}",
                        render_val(&previous),
                        render_val(&next)
                    )));
                }
                _ => items.push(self.parse_sum()?),
            }
        }
        if items.is_empty() {
            return Err(format!("{}: empty value fragment", self.file()));
        }
        if items.len() == 1 {
            Ok(items.pop().expect("one item"))
        } else {
            Ok(Val::List { items, sep: ' ' })
        }
    }

    fn parse_sum(&mut self) -> Result<Val, String> {
        let mut left = self.parse_product()?;
        loop {
            let Some(token) = self.peek() else {
                return Ok(left);
            };
            let Tok::Op { op, spaced_after } = token.tok else {
                return Ok(left);
            };
            if (op != '+' && op != '-') || !token.spaced_before || !spaced_after {
                return Ok(left);
            }
            self.index += 1;
            let right = self.parse_product()?;
            left = self.numeric_binary(op, &left, &right)?;
        }
    }

    fn parse_product(&mut self) -> Result<Val, String> {
        let mut left = self.parse_unary()?;
        loop {
            let Some(token) = self.peek() else {
                return Ok(left);
            };
            let Tok::Op { op, .. } = token.tok else {
                return Ok(left);
            };
            if op != '*' {
                return Ok(left);
            }
            self.index += 1;
            let right = self.parse_unary()?;
            left = self.numeric_binary('*', &left, &right)?;
        }
    }

    fn parse_unary(&mut self) -> Result<Val, String> {
        if matches!(self.peek().map(|token| &token.tok), Some(Tok::UnaryMinus)) {
            self.index += 1;
            let operand = self.parse_unary()?;
            return match operand {
                Val::Num { value, unit } => Ok(Val::Num {
                    value: -value,
                    unit,
                }),
                other => Err(format!(
                    "{}: unary `-` on the non-numeric value {:?}",
                    self.file(),
                    render_val(&other)
                )),
            };
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<Val, String> {
        let Some(token) = self.peek().cloned() else {
            return Err(format!("{}: expected a value", self.file()));
        };
        self.index += 1;
        match token.tok {
            Tok::Num { value, unit } => Ok(Val::Num { value, unit }),
            Tok::Ident(name) => Ok(Val::Lit(name)),
            Tok::Str(text) => Ok(Val::Lit(text)),
            Tok::Hash(text) => Ok(Val::Lit(text)),
            Tok::Bang(text) => Ok(Val::Lit(text)),
            Tok::Url(inner) => Ok(Val::Lit(self.rewrite_url(&inner))),
            Tok::Var { namespace, name } => {
                let raw = match namespace {
                    Some(namespace) => {
                        self.compiler
                            .lookup_namespaced_var(&namespace, &name, self.ctx)?
                    }
                    None => self
                        .ctx
                        .lookup(&name)
                        .map(str::to_string)
                        .ok_or_else(|| {
                            format!("{}: undefined variable `${name}`", self.file())
                        })?,
                };
                self.parse_stored_value(&raw)
            }
            Tok::Paren(inner) => {
                if find_colon(&inner).is_some() {
                    return Err(format!(
                        "{}: Sass map literals `({inner})` are not supported \
                         (construct: map literal)",
                        self.file()
                    ));
                }
                let value = self.compiler.eval_value(&inner, self.ctx)?;
                self.parse_stored_value(&value)
            }
            Tok::Call { name, args } => self.eval_function(&name, &args),
            Tok::Comma | Tok::Op { .. } | Tok::UnaryMinus => Err(format!(
                "{}: unexpected operator in value",
                self.file()
            )),
        }
    }

    /// Re-parses an already-evaluated value string (a variable's stored value
    /// or a parenthesized group's result) into a [`Val`].
    fn parse_stored_value(&self, raw: &str) -> Result<Val, String> {
        let raw = raw.trim();
        if raw.is_empty() {
            return Ok(Val::Lit(String::new()));
        }
        let tokens = tokenize_value(raw, &self.file())?;
        let mut parser = ValueParser {
            compiler: self.compiler,
            ctx: self.ctx,
            tokens: &tokens,
            index: 0,
        };
        let value = parser.parse_comma_list()?;
        if parser.index < tokens.len() {
            return Ok(Val::Lit(raw.to_string()));
        }
        Ok(value)
    }

    fn numeric_binary(&self, op: char, left: &Val, right: &Val) -> Result<Val, String> {
        let (Val::Num {
            value: left_value,
            unit: left_unit,
        }, Val::Num {
            value: right_value,
            unit: right_unit,
        }) = (left, right)
        else {
            return Err(format!(
                "{}: arithmetic `{}` needs numeric operands (found {:?} and {:?})",
                self.file(),
                op,
                render_val(left),
                render_val(right)
            ));
        };
        match op {
            '*' => {
                if !left_unit.is_empty() && !right_unit.is_empty() {
                    return Err(format!(
                        "{}: cannot multiply {}{} by {}{} (two dimensioned numbers)",
                        self.file(),
                        format_number(*left_value),
                        left_unit,
                        format_number(*right_value),
                        right_unit
                    ));
                }
                let unit = if left_unit.is_empty() {
                    right_unit.clone()
                } else {
                    left_unit.clone()
                };
                Ok(Val::Num {
                    value: left_value * right_value,
                    unit,
                })
            }
            '+' | '-' => {
                let unit = if left_unit.is_empty() {
                    right_unit.clone()
                } else if right_unit.is_empty() || left_unit.eq_ignore_ascii_case(right_unit) {
                    left_unit.clone()
                } else {
                    return Err(format!(
                        "{}: cannot combine {}{} and {}{} (incompatible units; use calc())",
                        self.file(),
                        format_number(*left_value),
                        left_unit,
                        format_number(*right_value),
                        right_unit
                    ));
                };
                let value = if op == '+' {
                    left_value + right_value
                } else {
                    left_value - right_value
                };
                Ok(Val::Num { value, unit })
            }
            _ => Err(format!("{}: unsupported operator `{op}`", self.file())),
        }
    }

    fn eval_function(&self, name: &str, args: &str) -> Result<Val, String> {
        let lower = name.to_ascii_lowercase();
        if SASS_ONLY_FUNCTIONS.contains(&lower.as_str()) {
            return Err(format!(
                "{}: the Sass function `{name}(...)` is not supported (construct: {name})",
                self.file()
            ));
        }
        match lower.as_str() {
            "calc" => {
                let result = self.eval_calc(args)?;
                Ok(match result {
                    Calc::Num { value, unit } => Val::Num { value, unit },
                    other => Val::Lit(format!("calc({})", render_calc_root(&other))),
                })
            }
            "min" | "max" | "clamp" => {
                // Arguments are calc expressions; simplify each, keep the
                // function.
                let mut rendered = Vec::new();
                for argument in split_top_level_commas(args) {
                    let argument = argument.trim();
                    if argument.is_empty() {
                        continue;
                    }
                    rendered.push(render_calc_root(&self.eval_calc(argument)?));
                }
                Ok(Val::Lit(format!("{name}({})", rendered.join(", "))))
            }
            "sqrt" => {
                let inner = self.compiler.eval_value(args, self.ctx)?;
                let value = self.parse_stored_value(&inner)?;
                match value {
                    Val::Num { value, unit } if unit.is_empty() => {
                        if value < 0.0 {
                            return Err(format!(
                                "{}: sqrt() of the negative number {}",
                                self.file(),
                                format_number(value)
                            ));
                        }
                        Ok(Val::Num {
                            value: value.sqrt(),
                            unit: String::new(),
                        })
                    }
                    other => Err(format!(
                        "{}: sqrt() needs a unitless number (found {:?})",
                        self.file(),
                        render_val(&other)
                    )),
                }
            }
            _ => {
                // A plain CSS function: evaluate each argument, keep the call.
                let mut rendered = Vec::new();
                for argument in split_top_level_commas(args) {
                    let argument = argument.trim();
                    if argument.is_empty() {
                        continue;
                    }
                    rendered.push(self.compiler.eval_value(argument, self.ctx)?);
                }
                Ok(Val::Lit(format!("{name}({})", rendered.join(", "))))
            }
        }
    }

    /// Rebases a relative `url(...)` written in an imported partial so it
    /// stays correct relative to the compilation root file (whose directory
    /// the downstream CSS pipeline resolves against).
    fn rewrite_url(&self, inner: &str) -> String {
        let raw = inner.trim();
        let unquoted = raw
            .strip_prefix(['"', '\''])
            .and_then(|value| value.strip_suffix(['"', '\'']))
            .unwrap_or(raw);
        let skip = unquoted.is_empty()
            || unquoted.starts_with('#')
            || unquoted.starts_with('/')
            || unquoted.starts_with("//")
            || unquoted.contains(':');
        if skip || self.ctx.file == self.compiler.root_file {
            return format!("url({raw})");
        }
        let split = unquoted.find(['?', '#']).unwrap_or(unquoted.len());
        let (path_part, suffix) = unquoted.split_at(split);
        let from_dir = self
            .ctx
            .file
            .parent()
            .unwrap_or_else(|| Path::new("."));
        let root_dir = self
            .compiler
            .root_file
            .parent()
            .unwrap_or_else(|| Path::new("."));
        let target = normalize_path(&from_dir.join(path_part));
        let rebased = relative_path(root_dir, &target);
        format!("url(\"{}{suffix}\")", rebased.display())
    }

    // -- calc() --------------------------------------------------------------

    fn eval_calc(&self, expression: &str) -> Result<Calc, String> {
        let tokens = tokenize_value(expression.trim(), &self.file())?;
        let mut parser = CalcParser {
            value: self,
            tokens: &tokens,
            index: 0,
        };
        let result = parser.parse_sum()?;
        if parser.index < tokens.len() {
            return Err(format!(
                "{}: cannot parse the calc() expression {expression:?}",
                self.file()
            ));
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// calc() simplification (dart-sass semantics).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Calc {
    Num { value: f64, unit: String },
    /// An unsimplifiable subexpression; `atom` means it needs no parentheses
    /// as a `*`/`/` operand (a function call or single token).
    Opaque { text: String, atom: bool },
    /// A flattened sum: `(sign, term)` where terms are Num or Opaque.
    Sum(Vec<(bool, Calc)>),
}

fn calc_terms(value: Calc) -> Vec<(bool, Calc)> {
    match value {
        Calc::Sum(terms) => terms,
        other => vec![(true, other)],
    }
}

/// Combines same-unit numeric terms; collapses a single-term positive sum.
fn simplify_sum(terms: Vec<(bool, Calc)>) -> Calc {
    let mut combined: Vec<(bool, Calc)> = Vec::new();
    for (positive, term) in terms {
        if let Calc::Num { value, unit } = &term {
            let signed = if positive { *value } else { -*value };
            if let Some((_, Calc::Num {
                value: existing,
                unit: existing_unit,
            })) = combined.iter_mut().find(|(sign, existing)| {
                *sign
                    && matches!(existing, Calc::Num { unit: existing_unit, .. }
                        if existing_unit.eq_ignore_ascii_case(unit))
            }) {
                *existing += signed;
                let _ = existing_unit;
                continue;
            }
            combined.push((
                true,
                Calc::Num {
                    value: signed,
                    unit: unit.clone(),
                },
            ));
            continue;
        }
        combined.push((positive, term));
    }
    if combined.len() == 1 && combined[0].0 {
        return combined.pop().expect("one term").1;
    }
    Calc::Sum(combined)
}

fn calc_add(left: Calc, right: Calc, subtract: bool) -> Calc {
    let mut terms = calc_terms(left);
    for (positive, term) in calc_terms(right) {
        terms.push((positive != subtract, term));
    }
    simplify_sum(terms)
}

fn render_calc_operand(value: &Calc) -> String {
    match value {
        Calc::Num { value, unit } => format!("{}{unit}", format_number(*value)),
        Calc::Opaque { text, atom } => {
            if *atom {
                text.clone()
            } else {
                format!("({text})")
            }
        }
        Calc::Sum(_) => format!("({})", render_calc_root(value)),
    }
}

fn render_calc_root(value: &Calc) -> String {
    match value {
        Calc::Num { value, unit } => format!("{}{unit}", format_number(*value)),
        Calc::Opaque { text, .. } => text.clone(),
        Calc::Sum(terms) => {
            let mut out = String::new();
            for (index, (positive, term)) in terms.iter().enumerate() {
                // A combined numeric term carries its sign in `value`; fold a
                // negative value into the ` - ` operator so the output reads
                // `50vh - 35.5vmin`, never `50vh + -35.5vmin`.
                let (effective_positive, rendered) = match term {
                    Calc::Num { value, unit } if index > 0 => (
                        *positive == (*value >= 0.0),
                        format!("{}{unit}", format_number(value.abs())),
                    ),
                    Calc::Num { value, unit } => {
                        (*positive, format!("{}{unit}", format_number(*value)))
                    }
                    other => (*positive, render_calc_operand(other)),
                };
                if index == 0 {
                    if !effective_positive {
                        out.push_str("-1 * ");
                    }
                    out.push_str(&rendered);
                } else {
                    out.push_str(if effective_positive { " + " } else { " - " });
                    out.push_str(&rendered);
                }
            }
            out
        }
    }
}

struct CalcParser<'a> {
    value: &'a ValueParser<'a>,
    tokens: &'a [SpacedTok],
    index: usize,
}

impl CalcParser<'_> {
    fn file(&self) -> String {
        self.value.file()
    }

    fn peek(&self) -> Option<&SpacedTok> {
        self.tokens.get(self.index)
    }

    fn parse_sum(&mut self) -> Result<Calc, String> {
        let mut left = self.parse_product()?;
        loop {
            let Some(token) = self.peek() else {
                return Ok(left);
            };
            let Tok::Op { op, .. } = token.tok else {
                return Ok(left);
            };
            if op != '+' && op != '-' {
                return Ok(left);
            }
            self.index += 1;
            let right = self.parse_product()?;
            left = calc_add(left, right, op == '-');
        }
    }

    fn parse_product(&mut self) -> Result<Calc, String> {
        let mut left = self.parse_unary()?;
        loop {
            let Some(token) = self.peek() else {
                return Ok(left);
            };
            let Tok::Op { op, .. } = token.tok else {
                return Ok(left);
            };
            if op != '*' && op != '/' {
                return Ok(left);
            }
            self.index += 1;
            let right = self.parse_unary()?;
            left = self.combine_product(op, left, right)?;
        }
    }

    fn combine_product(&self, op: char, left: Calc, right: Calc) -> Result<Calc, String> {
        if let (Calc::Num {
            value: left_value,
            unit: left_unit,
        }, Calc::Num {
            value: right_value,
            unit: right_unit,
        }) = (&left, &right)
        {
            if op == '*' {
                if !left_unit.is_empty() && !right_unit.is_empty() {
                    return Err(format!(
                        "{}: calc() cannot multiply two dimensioned numbers \
                         ({left_unit} * {right_unit})",
                        self.file()
                    ));
                }
                let unit = if left_unit.is_empty() {
                    right_unit.clone()
                } else {
                    left_unit.clone()
                };
                return Ok(Calc::Num {
                    value: left_value * right_value,
                    unit,
                });
            }
            // Division: the divisor must be a unitless, non-zero number.
            if right_unit.is_empty() {
                if *right_value == 0.0 {
                    return Err(format!("{}: division by zero in calc()", self.file()));
                }
                return Ok(Calc::Num {
                    value: left_value / right_value,
                    unit: left_unit.clone(),
                });
            }
            return Err(format!(
                "{}: calc() cannot divide by a dimensioned number ({right_unit})",
                self.file()
            ));
        }
        // Symbolic: keep the operation without distributing (dart-sass keeps
        // `(100vw - 60vmin) / 2` intact).
        Ok(Calc::Opaque {
            text: format!(
                "{} {op} {}",
                render_calc_operand(&left),
                render_calc_operand(&right)
            ),
            atom: false,
        })
    }

    fn parse_unary(&mut self) -> Result<Calc, String> {
        let negate = match self.peek().map(|token| &token.tok) {
            Some(Tok::UnaryMinus) => {
                self.index += 1;
                true
            }
            _ => false,
        };
        let operand = self.parse_atom()?;
        if !negate {
            return Ok(operand);
        }
        Ok(match operand {
            Calc::Num { value, unit } => Calc::Num {
                value: -value,
                unit,
            },
            Calc::Sum(terms) => simplify_sum(
                terms
                    .into_iter()
                    .map(|(positive, term)| (!positive, term))
                    .collect(),
            ),
            opaque => Calc::Opaque {
                text: format!("-1 * {}", render_calc_operand(&opaque)),
                atom: false,
            },
        })
    }

    fn parse_atom(&mut self) -> Result<Calc, String> {
        let Some(token) = self.peek().cloned() else {
            return Err(format!("{}: expected a calc() operand", self.file()));
        };
        self.index += 1;
        match token.tok {
            Tok::Num { value, unit } => Ok(Calc::Num { value, unit }),
            Tok::Paren(inner) => self.value.eval_calc(&inner),
            Tok::Var { namespace, name } => {
                let raw = match namespace {
                    Some(namespace) => self
                        .value
                        .compiler
                        .lookup_namespaced_var(&namespace, &name, self.value.ctx)?,
                    None => self
                        .value
                        .ctx
                        .lookup(&name)
                        .map(str::to_string)
                        .ok_or_else(|| {
                            format!("{}: undefined variable `${name}`", self.file())
                        })?,
                };
                self.value.eval_calc(&raw)
            }
            Tok::Call { name, args } => {
                let lower = name.to_ascii_lowercase();
                if SASS_ONLY_FUNCTIONS.contains(&lower.as_str()) {
                    return Err(format!(
                        "{}: the Sass function `{name}(...)` is not supported \
                         (construct: {name})",
                        self.file()
                    ));
                }
                match lower.as_str() {
                    "calc" => self.value.eval_calc(&args),
                    "sqrt" => match self.value.eval_function("sqrt", &args)? {
                        Val::Num { value, unit } => Ok(Calc::Num { value, unit }),
                        _ => unreachable!("sqrt always yields a number"),
                    },
                    _ => {
                        let rendered = self.value.eval_function(&name, &args)?;
                        Ok(Calc::Opaque {
                            text: render_val(&rendered),
                            atom: true,
                        })
                    }
                }
            }
            Tok::Ident(name) => Ok(Calc::Opaque {
                text: name,
                atom: true,
            }),
            other => Err(format!(
                "{}: unsupported token {other:?} in calc()",
                self.file()
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Path helpers (url rebasing for imported partials).
// ---------------------------------------------------------------------------

/// Lexically normalizes `.` / `..` components (no filesystem access).
fn normalize_path(path: &Path) -> PathBuf {
    let mut parts: Vec<std::path::Component> = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if matches!(parts.last(), Some(std::path::Component::Normal(_))) {
                    parts.pop();
                } else {
                    parts.push(component);
                }
            }
            other => parts.push(other),
        }
    }
    parts.iter().collect()
}

/// The relative path from directory `from` to `to` (both absolute).
fn relative_path(from: &Path, to: &Path) -> PathBuf {
    let from = normalize_path(from);
    let to = normalize_path(to);
    let from_components: Vec<_> = from.components().collect();
    let to_components: Vec<_> = to.components().collect();
    let common = from_components
        .iter()
        .zip(&to_components)
        .take_while(|(left, right)| left == right)
        .count();
    let mut result = PathBuf::new();
    for _ in common..from_components.len() {
        result.push("..");
    }
    for component in &to_components[common..] {
        result.push(component);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compile(source: &str) -> String {
        compile_scss(Path::new("/project/test.scss"), source, &ScssOptions::default())
            .expect("compile should succeed")
            .css
    }

    fn compile_error(source: &str) -> String {
        compile_scss(Path::new("/project/test.scss"), source, &ScssOptions::default())
            .expect_err("compile should fail")
    }

    /// Collapses the pretty-printed output for terse assertions.
    fn flat(css: &str) -> String {
        css.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    // -- variables -----------------------------------------------------------

    #[test]
    fn variable_declaration_and_reference() {
        let css = compile("$pad: 12px;\n.a { padding: $pad; }\n");
        assert_eq!(flat(&css), ".a { padding: 12px; }");
    }

    #[test]
    fn variable_referencing_variable() {
        let css = compile("$a: #fff;\n$b: $a;\n.x { color: $b; }\n");
        assert_eq!(flat(&css), ".x { color: #fff; }");
    }

    #[test]
    fn variable_rule_local_scope() {
        let css = compile(
            "$w: 1px;\n.a { $w: 60vmin; width: $w; }\n.b { width: $w; }\n",
        );
        assert_eq!(flat(&css), ".a { width: 60vmin; } .b { width: 60vmin; }");
    }

    #[test]
    fn block_local_variable_declaration() {
        // A variable FIRST declared inside a rule stays local to it.
        let css = compile(".a { $w: 5px; width: $w; }\n.b { height: 1px; }\n");
        assert_eq!(flat(&css), ".a { width: 5px; } .b { height: 1px; }");
        let error = compile_error(".a { $w: 5px; }\n.b { width: $w; }\n");
        assert!(error.contains("undefined variable `$w`"), "{error}");
    }

    #[test]
    fn variable_default_flag() {
        let css = compile("$x: 1px;\n$x: 2px !default;\n.a { width: $x; }\n");
        assert_eq!(flat(&css), ".a { width: 1px; }");
        let css = compile("$y: 3px !default;\n.a { width: $y; }\n");
        assert_eq!(flat(&css), ".a { width: 3px; }");
    }

    #[test]
    fn undefined_variable_is_an_error() {
        let error = compile_error(".a { width: $missing; }\n");
        assert!(error.contains("undefined variable `$missing`"), "{error}");
    }

    #[test]
    fn variable_list_value_passes_through() {
        let css = compile("$m: 0 auto;\n.a { margin: $m; }\n");
        assert_eq!(flat(&css), ".a { margin: 0 auto; }");
    }

    // -- nesting and `&` -----------------------------------------------------

    #[test]
    fn simple_nesting() {
        let css = compile("ol { margin: 0; li { margin-bottom: 14px; } }\n");
        assert_eq!(
            flat(&css),
            "ol { margin: 0; } ol li { margin-bottom: 14px; }"
        );
    }

    #[test]
    fn ampersand_pseudo_and_class() {
        let css = compile(
            "button { color: red; &:hover { color: blue; } &.primary { color: green; } }\n",
        );
        assert_eq!(
            flat(&css),
            "button { color: red; } button:hover { color: blue; } \
             button.primary { color: green; }"
        );
    }

    #[test]
    fn ampersand_parent_position() {
        let css = compile(".child { .parent & { color: red; } }\n");
        assert_eq!(flat(&css), ".parent .child { color: red; }");
    }

    #[test]
    fn nested_selector_lists_cross_multiply() {
        let css = compile(".a, .b { &:hover, .x & { color: red; } }\n");
        assert_eq!(
            flat(&css),
            ".a:hover, .x .a, .b:hover, .x .b { color: red; }"
        );
    }

    #[test]
    fn deep_nesting_with_child_combinator() {
        let css = compile(".menu { > div { span { color: red; } } }\n");
        assert_eq!(flat(&css), ".menu > div span { color: red; }");
    }

    #[test]
    fn double_ampersand_compound() {
        let css = compile(".a, .b { &.a { z-index: 2; } }\n");
        assert_eq!(flat(&css), ".a.a, .b.a { z-index: 2; }");
    }

    #[test]
    fn empty_rules_are_dropped() {
        let css = compile(".a { }\n.b { color: red; }\n");
        assert_eq!(flat(&css), ".b { color: red; }");
    }

    // -- @media bubbling -----------------------------------------------------

    #[test]
    fn nested_media_bubbles_with_parent_selector() {
        let css = compile(
            "$bp: 800px;\n#title { flex: 1; @media only screen and (min-width: $bp) \
             { flex: inherit; } }\n",
        );
        assert_eq!(
            flat(&css),
            "#title { flex: 1; } @media only screen and (min-width: 800px) \
             { #title { flex: inherit; } }"
        );
    }

    #[test]
    fn media_with_nested_rule_inside_rule() {
        let css = compile(
            ".wrap { @media (min-width: 10px) { .inner { color: red; } } }\n",
        );
        assert_eq!(
            flat(&css),
            "@media (min-width: 10px) { .wrap .inner { color: red; } }"
        );
    }

    #[test]
    fn empty_nested_media_is_dropped() {
        let css = compile(".a { flex: 1; @media (min-width: 5px) { } display: flex; }\n");
        assert_eq!(flat(&css), ".a { flex: 1; display: flex; }");
    }

    #[test]
    fn top_level_media_with_rules() {
        let css = compile("@media (max-width: 425px) { html { font-size: 8px; } }\n");
        assert_eq!(
            flat(&css),
            "@media (max-width: 425px) { html { font-size: 8px; } }"
        );
    }

    // -- mixins --------------------------------------------------------------

    #[test]
    fn mixin_without_arguments() {
        let css = compile(
            "@mixin heading { color: red; font-size: 5rem; }\n\
             h1 { @include heading; margin: 0; }\n",
        );
        assert_eq!(
            flat(&css),
            "h1 { color: red; font-size: 5rem; margin: 0; }"
        );
    }

    #[test]
    fn mixin_with_arguments_and_defaults() {
        let css = compile(
            "@mixin box($w, $h: 10px) { width: $w; height: $h; }\n\
             .a { @include box(5px); }\n.b { @include box(1px, 2px); }\n",
        );
        assert_eq!(
            flat(&css),
            ".a { width: 5px; height: 10px; } .b { width: 1px; height: 2px; }"
        );
    }

    #[test]
    fn mixin_with_nested_rules_and_includes() {
        let css = compile(
            "@mixin inner { color: red; }\n\
             @mixin outer { margin: 0; > div { @include inner; } }\n\
             .menu { @include outer; }\n",
        );
        assert_eq!(
            flat(&css),
            ".menu { margin: 0; } .menu > div { color: red; }"
        );
    }

    #[test]
    fn mixin_uses_defining_scope_variables() {
        let css = compile(
            "$clr: #123;\n@mixin tint { color: $clr; }\n.a { @include tint; }\n",
        );
        assert_eq!(flat(&css), ".a { color: #123; }");
    }

    #[test]
    fn mixin_local_variable() {
        let css = compile(
            "$scale: 6rem;\n@mixin stripes { $size: $scale * 2; background-size: $size $size; }\n\
             body { @include stripes; }\n",
        );
        assert_eq!(
            flat(&css),
            "body { background-size: 12rem 12rem; }"
        );
    }

    #[test]
    fn unknown_mixin_is_an_error() {
        let error = compile_error(".a { @include missing; }\n");
        assert!(error.contains("mixin `missing` is not defined"), "{error}");
    }

    #[test]
    fn keyword_arguments_are_rejected() {
        let error = compile_error(
            "@mixin m($a: 1px) { width: $a; }\n.x { @include m($a: 2px); }\n",
        );
        assert!(error.contains("keyword arguments"), "{error}");
    }

    // -- arithmetic and functions -------------------------------------------

    #[test]
    fn multiplication_outside_calc() {
        let css = compile("$pad: 12px;\n.a { margin-top: 2 * $pad; }\n");
        assert_eq!(flat(&css), ".a { margin-top: 24px; }");
    }

    #[test]
    fn unary_minus_on_variable() {
        let css = compile("$s: 20vh;\n.a { top: -$s * 3; }\n");
        assert_eq!(flat(&css), ".a { top: -60vh; }");
    }

    #[test]
    fn sqrt_matches_dart_sass_precision() {
        let css = compile("$s: 6rem;\n.a { background-size: $s * sqrt(8); }\n");
        assert_eq!(flat(&css), ".a { background-size: 16.9705627485rem; }");
    }

    #[test]
    fn negative_number_literals_stay_list_items() {
        let css = compile(".a { margin: 0 -2px; text-shadow: 0 0.3rem 0.6rem black; }\n");
        assert_eq!(
            flat(&css),
            ".a { margin: 0 -2px; text-shadow: 0 0.3rem 0.6rem black; }"
        );
    }

    #[test]
    fn spaced_binary_subtraction() {
        let css = compile("$a: 10px;\n.x { width: $a - 3px; }\n");
        assert_eq!(flat(&css), ".x { width: 7px; }");
    }

    #[test]
    fn incompatible_units_error_outside_calc() {
        let error = compile_error(".a { width: 10px + 2vh; }\n");
        assert!(error.contains("incompatible units"), "{error}");
    }

    #[test]
    fn sass_only_function_is_rejected() {
        let error = compile_error(".a { color: darken(#fff, 10%); }\n");
        assert!(error.contains("`darken(...)` is not supported"), "{error}");
    }

    #[test]
    fn namespaced_function_is_rejected() {
        let error = compile_error(".a { width: math.div(10px, 2); }\n");
        assert!(error.contains("math.div"), "{error}");
    }

    #[test]
    fn css_functions_pass_through_with_evaluated_arguments() {
        let css = compile(
            "$c: #7f5656;\n.a { background-image: linear-gradient(-45deg, $c 0, $c 25%, \
             transparent 25%, transparent 100%); }\n",
        );
        assert_eq!(
            flat(&css),
            ".a { background-image: linear-gradient(-45deg, #7f5656 0, #7f5656 25%, \
             transparent 25%, transparent 100%); }"
        );
    }

    #[test]
    fn slash_stays_literal_outside_calc() {
        let css = compile(".a { font: 12px/1.5 serif; aspect-ratio: 16/9; }\n");
        assert_eq!(
            flat(&css),
            ".a { font: 12px/1.5 serif; aspect-ratio: 16/9; }"
        );
    }

    #[test]
    fn important_flag_passes_through() {
        let css = compile(".a { color: red !important; }\n");
        assert_eq!(flat(&css), ".a { color: red !important; }");
    }

    // -- calc() --------------------------------------------------------------

    #[test]
    fn calc_single_value_drops_wrapper() {
        let css = compile("$g: 65vmin;\n.a { width: calc($g); }\n");
        assert_eq!(flat(&css), ".a { width: 65vmin; }");
    }

    #[test]
    fn calc_same_unit_arithmetic_simplifies() {
        let css = compile(
            "$g: 65vmin;\n$t: 3vmin;\n.a { width: calc($g + $t * 2); \
             height: calc(100vh + 20vh * 2); }\n",
        );
        assert_eq!(flat(&css), ".a { width: 71vmin; height: 140vh; }");
    }

    #[test]
    fn calc_division_simplifies() {
        let css = compile("$g: 65vmin;\n.a { width: calc($g / 8 * 0.7); }\n");
        assert_eq!(flat(&css), ".a { width: 5.6875vmin; }");
    }

    #[test]
    fn calc_mixed_units_combine_same_unit_terms() {
        let css = compile(
            "$g: 65vmin;\n$t: 3vmin;\n.a { top: calc(50vh - $g / 2 - $t); }\n",
        );
        assert_eq!(flat(&css), ".a { top: calc(50vh - 35.5vmin); }");
    }

    #[test]
    fn calc_symbolic_division_keeps_structure() {
        let css = compile("$w: 60vmin;\n.a { left: calc((100vw - $w) / 2); }\n");
        assert_eq!(flat(&css), ".a { left: calc((100vw - 60vmin) / 2); }");
    }

    #[test]
    fn calc_parenthesized_negation() {
        let css = compile("$z: 20vh;\n.a { top: calc(-100vh - ($z * 4)); }\n");
        assert_eq!(flat(&css), ".a { top: -180vh; }");
    }

    #[test]
    fn calc_percent_mixed_stays() {
        let css = compile("$r: 4vmin;\n.a { width: calc(100% - $r / 2); }\n");
        assert_eq!(flat(&css), ".a { width: calc(100% - 2vmin); }");
    }

    #[test]
    fn calc_with_var_reference_stays_opaque() {
        let css = compile(".a { width: calc(var(--w) / 2); }\n");
        assert_eq!(flat(&css), ".a { width: calc(var(--w) / 2); }");
    }

    #[test]
    fn calc_division_by_zero_is_an_error() {
        let error = compile_error(".a { width: calc(10px / 0); }\n");
        assert!(error.contains("division by zero"), "{error}");
    }

    #[test]
    fn min_max_args_simplify() {
        let css = compile("$a: 30vmin;\n.a { width: max($a * 2, 60vw); }\n");
        assert_eq!(flat(&css), ".a { width: max(60vmin, 60vw); }");
    }

    // -- keyframes / font-face ----------------------------------------------

    #[test]
    fn keyframes_with_variables_in_frames() {
        let css = compile(
            "$s: 6rem;\n@keyframes move { 0% { background-position: 0; } \
             100% { background-position: $s * 2; } }\n",
        );
        assert_eq!(
            flat(&css),
            "@keyframes move { 0% { background-position: 0; } \
             100% { background-position: 12rem; } }"
        );
    }

    #[test]
    fn keyframe_frame_selector_lists() {
        let css = compile(
            "@keyframes shake { 0%,\n 100% { transform: scale(1); } \
             50% { transform: scale(1.5); } }\n",
        );
        assert_eq!(
            flat(&css),
            "@keyframes shake { 0%, 100% { transform: scale(1); } \
             50% { transform: scale(1.5); } }"
        );
    }

    #[test]
    fn font_face_block() {
        let css = compile(
            "@font-face { font-family: \"ConcertOne\"; \
             src: url(\"/fonts/ConcertOne-Regular.ttf\") format(\"truetype\"); }\n",
        );
        assert_eq!(
            flat(&css),
            "@font-face { font-family: \"ConcertOne\"; \
             src: url(\"/fonts/ConcertOne-Regular.ttf\") format(\"truetype\"); }"
        );
    }

    #[test]
    fn multiline_value_with_silent_comments() {
        let css = compile(
            ".a {\n  src: local(''),\n      // Chrome 26+\n      url('./f.woff2') \
             format('woff2'),\n      // old\n      url('./f.woff') format('woff');\n}\n",
        );
        assert_eq!(
            flat(&css),
            ".a { src: local(''), url('./f.woff2') format('woff2'), \
             url('./f.woff') format('woff'); }"
        );
    }

    #[test]
    fn data_uri_with_slashes_survives() {
        let css = compile(
            ".a { cursor: url(\"data:image/svg+xml;base64,PHN2//Zz4K+A==\"), auto; }\n",
        );
        assert_eq!(
            flat(&css),
            ".a { cursor: url(\"data:image/svg+xml;base64,PHN2//Zz4K+A==\"), auto; }"
        );
    }

    #[test]
    fn custom_properties_stay_verbatim() {
        let css = compile(".a { --my-var: 10px 20px; width: var(--my-var, 5px); }\n");
        assert_eq!(
            flat(&css),
            ".a { --my-var: 10px 20px; width: var(--my-var, 5px); }"
        );
    }

    // -- hard errors ---------------------------------------------------------

    #[test]
    fn extend_is_rejected() {
        let error = compile_error(".a { @extend .b; }\n");
        assert!(error.contains("@extend"), "{error}");
    }

    #[test]
    fn control_flow_is_rejected() {
        for source in [
            "@if true { .a { color: red; } }\n",
            "@each $i in 1 2 { .a { width: $i; } }\n",
            "@for $i from 1 through 3 { .a { width: $i; } }\n",
            "@function f() { @return 1; }\n",
        ] {
            let error = compile_error(source);
            assert!(error.contains("is not supported"), "{error}");
        }
    }

    #[test]
    fn interpolation_is_rejected() {
        let error = compile_error("$n: 5;\n.a-#{$n} { color: red; }\n");
        assert!(error.contains("interpolation"), "{error}");
    }

    #[test]
    fn placeholder_selector_is_rejected() {
        let error = compile_error("%base { color: red; }\n");
        assert!(error.contains("placeholder"), "{error}");
    }

    #[test]
    fn use_with_configuration_is_rejected() {
        let error = compile_error("@use './theme' with ($x: 1);\n");
        assert!(error.contains("with (...)"), "{error}");
    }

    #[test]
    fn builtin_sass_module_is_rejected() {
        let error = compile_error("@use 'sass:math';\n.a { width: 1px; }\n");
        assert!(error.contains("sass:math"), "{error}");
    }

    #[test]
    fn map_literal_is_rejected() {
        let error = compile_error("$m: (a: 1);\n.x { width: $m; }\n");
        assert!(error.contains("map literal"), "{error}");
    }

    #[test]
    fn unknown_at_rule_is_rejected() {
        let error = compile_error("@unknown-thing { }\n");
        assert!(error.contains("@unknown-thing"), "{error}");
    }

    // -- @use / @import with real files --------------------------------------

    fn write(dir: &Path, name: &str, contents: &str) -> PathBuf {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn use_with_namespace_variables_and_mixins() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "theme.scss",
            "$clr: #8a5c5d;\n@mixin breathing { animation: breathing 3s infinite; }\n\
             body { color: $clr; }\n",
        );
        let root = write(
            dir.path(),
            "page.module.scss",
            "@use \"./theme.scss\" as theme;\n\
             .logo { color: theme.$clr; @include theme.breathing; }\n",
        );
        let source = fs::read_to_string(&root).unwrap();
        let compiled =
            compile_scss(&root, &source, &ScssOptions::default()).expect("compiles");
        assert_eq!(
            flat(&compiled.css),
            "body { color: #8a5c5d; } .logo { color: #8a5c5d; \
             animation: breathing 3s infinite; }"
        );
        assert_eq!(compiled.loaded_files.len(), 1);
        assert!(
            compiled.loaded_files[0].ends_with("theme.scss"),
            "{:?}",
            compiled.loaded_files
        );
    }

    #[test]
    fn use_default_namespace_and_partial_convention() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "_vars.scss", "$pad: 12px;\n");
        let root = write(
            dir.path(),
            "app.scss",
            "@use './vars';\n.a { padding: vars.$pad; }\n",
        );
        let source = fs::read_to_string(&root).unwrap();
        let compiled =
            compile_scss(&root, &source, &ScssOptions::default()).expect("compiles");
        assert_eq!(flat(&compiled.css), ".a { padding: 12px; }");
    }

    #[test]
    fn use_star_merges_members() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "_vars.scss", "$pad: 7px;\n");
        let root = write(
            dir.path(),
            "app.scss",
            "@use './vars' as *;\n.a { padding: $pad; }\n",
        );
        let source = fs::read_to_string(&root).unwrap();
        let compiled =
            compile_scss(&root, &source, &ScssOptions::default()).expect("compiles");
        assert_eq!(flat(&compiled.css), ".a { padding: 7px; }");
    }

    #[test]
    fn root_relative_use_resolves_against_project_root() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "src/global/theme.scss", "$z: 90;\n");
        let root = write(
            dir.path(),
            "src/deep/page.module.scss",
            "@use \"/src/global/theme.scss\" as theme;\n.hud { z-index: theme.$z; }\n",
        );
        let source = fs::read_to_string(&root).unwrap();
        let options = ScssOptions {
            additional_data: None,
            root: Some(dir.path().to_path_buf()),
        };
        let compiled = compile_scss(&root, &source, &options).expect("compiles");
        assert_eq!(flat(&compiled.css), ".hud { z-index: 90; }");
    }

    #[test]
    fn additional_data_prepends_to_root() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "src/global/theme.scss", "$clr: #e6a459;\n");
        let root = write(
            dir.path(),
            "src/x.module.scss",
            ".a { color: theme.$clr; }\n",
        );
        let source = fs::read_to_string(&root).unwrap();
        let options = ScssOptions {
            additional_data: Some(
                "@use \"/src/global/theme.scss\" as theme;".to_string(),
            ),
            root: Some(dir.path().to_path_buf()),
        };
        let compiled = compile_scss(&root, &source, &options).expect("compiles");
        assert_eq!(flat(&compiled.css), ".a { color: #e6a459; }");
    }

    #[test]
    fn used_module_css_emitted_once_before_user_css() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "_base.scss", "body { margin: 0; }\n");
        let root = write(
            dir.path(),
            "app.scss",
            "@use './base';\n@use './base' as b;\n.a { color: red; }\n",
        );
        let source = fs::read_to_string(&root).unwrap();
        let compiled =
            compile_scss(&root, &source, &ScssOptions::default()).expect("compiles");
        assert_eq!(
            flat(&compiled.css),
            "body { margin: 0; } .a { color: red; }"
        );
    }

    #[test]
    fn sass_import_evaluates_in_importer_scope() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "_shared.scss", "$gap: 4px;\n.shared { margin: $gap; }\n");
        let root = write(
            dir.path(),
            "app.scss",
            "@import './shared';\n.own { padding: $gap; }\n",
        );
        let source = fs::read_to_string(&root).unwrap();
        let compiled =
            compile_scss(&root, &source, &ScssOptions::default()).expect("compiles");
        assert_eq!(
            flat(&compiled.css),
            ".shared { margin: 4px; } .own { padding: 4px; }"
        );
        assert_eq!(compiled.loaded_files.len(), 1);
    }

    #[test]
    fn imported_partial_relative_urls_are_rebased() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "nested/_fonts.scss",
            ".f { background: url('img/a.png'); }\n",
        );
        let root = write(dir.path(), "app.scss", "@import './nested/fonts';\n");
        let source = fs::read_to_string(&root).unwrap();
        let compiled =
            compile_scss(&root, &source, &ScssOptions::default()).expect("compiles");
        assert_eq!(
            flat(&compiled.css),
            ".f { background: url(\"nested/img/a.png\"); }"
        );
    }

    #[test]
    fn css_import_passes_through() {
        let css = compile("@import './plain.css';\n.a { color: red; }\n");
        assert_eq!(
            flat(&css),
            "@import './plain.css'; .a { color: red; }"
        );
    }

    #[test]
    fn missing_use_target_names_candidates() {
        let error = compile_error("@use './nope';\n.a { color: red; }\n");
        assert!(error.contains("cannot resolve"), "{error}");
        assert!(error.contains("_nope.scss"), "{error}");
    }

    #[test]
    fn module_loop_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "a.scss", "@use './b';\n");
        write(dir.path(), "b.scss", "@use './a';\n");
        let root = dir.path().join("a.scss");
        let source = fs::read_to_string(&root).unwrap();
        let error = compile_scss(&root, &source, &ScssOptions::default())
            .expect_err("loop should fail");
        assert!(error.contains("loop"), "{error}");
    }

    #[test]
    fn indented_sass_syntax_is_rejected() {
        let error = compile_scss(
            Path::new("/project/x.sass"),
            "body\n  color: red\n",
            &ScssOptions::default(),
        )
        .expect_err("indented syntax should fail");
        assert!(error.contains("indented"), "{error}");
    }

    // -- number formatting ---------------------------------------------------

    #[test]
    fn number_formatting_matches_dart_sass() {
        assert_eq!(format_number(16.97056274847714), "16.9705627485");
        assert_eq!(format_number(24.0), "24");
        assert_eq!(format_number(0.5), "0.5");
        assert_eq!(format_number(-0.0), "0");
        assert_eq!(format_number(5.6875), "5.6875");
    }
}
