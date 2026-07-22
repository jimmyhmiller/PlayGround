//! Native Tailwind v4 CSS compilation.
//!
//! A Tailwind v4 entry (`@import 'tailwindcss'`, optional `@layer base` with
//! `@apply`, plus the app's own plain CSS) is compiled by the reference Vite
//! build (`@tailwindcss/vite`) into a single extracted stylesheet: the v4
//! preflight, the theme tokens the app references, the app's own rules, and one
//! utility rule per class the app uses. Diffpack used to copy the raw source
//! through, so the browser fetched `@import 'tailwindcss'` and 404'd.
//!
//! This module is a native Rust implementation of that compile. It is a *general*
//! utility engine driven by faithful Tailwind v4 reference data — the published
//! default theme ([`THEME_CSS`], verbatim from the `tailwindcss` package) and the
//! resolved preflight ([`PREFLIGHT_CSS`]) — never a lookup table of an app's
//! specific classes. A class the app uses that the engine does not yet handle is
//! a hard, specific error naming the token; it is never silently dropped.
//!
//! Candidate scanning is precision-scoped (unlike Tailwind's scan-every-string
//! heuristic): it extracts the string values that *flow into* `class`/`className`
//! attributes — literals, template literals, ternary branches, and `const`
//! string bindings referenced from those positions — so every candidate is a real
//! class the app applies and an unhandled one is a genuine gap.
//!
//! Color opacity modifiers (`bg-black/30`) compile to the modern
//! `color-mix(in oklab, …)` declaration Tailwind emits; the static sRGB fallback
//! hex the reference minifier additionally inlines for pre-`color-mix` browsers
//! is not duplicated (every target browser resolves the color-mix branch).
//!
//! The compile is a build-emit step (it runs once per `emit`, like manifest
//! generation), not part of the incremental transform hot path: a leaf edit still
//! re-transforms exactly one module.

use std::collections::{BTreeMap, BTreeSet};

/// The published Tailwind v4.3.3 default theme, verbatim from
/// `node_modules/tailwindcss/theme.css`. Parsed for the token values the app's
/// utilities reference (colors, spacing, font sizes/weights, radii, fonts).
const THEME_CSS: &str = include_str!("tailwind_theme.css");

/// The resolved Tailwind v4.3.3 preflight (base reset). Identical for every v4
/// app — faithful reference data, not app-specific.
const PREFLIGHT_CSS: &str = include_str!("tailwind_preflight.css");

const VERSION_BANNER: &str =
    "/*! tailwindcss v4.3.3 | MIT License | https://tailwindcss.com */";

/// The `@supports` feature query Tailwind v4 guards its registered-property
/// fallbacks with (for browsers without `@property`).
const PROPERTIES_SUPPORTS: &str = "@supports (((-webkit-hyphens:none)) and (not (margin-trim:inline))) or ((-moz-orient:inline) and (not (color:rgb(from red r g b))))";

/// The full `box-shadow` composition every shadow/ring utility assigns, verbatim
/// from Tailwind v4.
const BOX_SHADOW_CHAIN: &str = "var(--tw-inset-shadow), var(--tw-inset-ring-shadow), var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow)";

/// Utility rules grouped for output: `(variant order, media key)` ->
/// `(family rank, class, rule css)` entries, sorted within each group.
type RuleGroups = BTreeMap<(u8, String), Vec<(u16, String, String)>>;

/// Whether a CSS source is a Tailwind v4 entry point, i.e. it imports the
/// framework via `@import 'tailwindcss'` (single or double quotes, with or
/// without a `source(...)` argument).
pub fn is_tailwind_entry(css: &str) -> bool {
    css.lines().any(|line| {
        let line = line.trim();
        line.starts_with("@import")
            && (line.contains("'tailwindcss'") || line.contains("\"tailwindcss\""))
    })
}

/// Compiles a Tailwind v4 CSS entry into a plain, self-contained stylesheet.
///
/// `candidate_classes` are the utility class tokens scanned from the app's source
/// (see [`scan_class_candidates`]). Every candidate must resolve to a utility —
/// or to a class the app's own CSS defines — or this returns a hard error naming
/// every unresolved token.
pub fn compile(css: &str, candidate_classes: &BTreeSet<String>) -> Result<String, String> {
    let theme = Theme::parse(THEME_CSS);
    let mut tw_props: BTreeSet<TwProp> = BTreeSet::new();

    // 1. Process the app's own CSS first: strip the framework import, expand
    //    `@apply` inside `@layer base`, pass plain (unlayered) rules through,
    //    and learn which class names the app's own CSS defines.
    let user = process_user_css(css, &theme, &mut tw_props)?;

    // 2. Generate one rule per candidate utility, grouped by variant order and
    //    media wrapper. Collect every failure into one hard error.
    let mut groups: RuleGroups = BTreeMap::new();
    let mut errors: Vec<String> = Vec::new();
    for class in candidate_classes {
        // Classes the app's own stylesheet defines are satisfied there; `group`
        // and `peer` are Tailwind marker classes that generate no CSS.
        if user.defined_classes.contains(class) || is_marker_class(class) {
            continue;
        }
        match render_utility(class, &theme, &mut tw_props, &user.custom_variants) {
            Ok(rule) => groups
                .entry((rule.order, rule.media_key))
                .or_default()
                .push((rule.rank, class.clone(), rule.css)),
            // Tailwind's own scanner treats EVERY string token in the source
            // tree as a candidate and silently ignores the ones that are not
            // utilities at all (`zero`, `data`, ...). Matching that: a token
            // whose root no utility family recognizes is skipped; a token with
            // a RECOGNIZED root but an unsupported form stays a hard error —
            // that is the surface where silence would ship a broken style.
            Err(error) => {
                if utility_root_recognized(class) {
                    errors.push(error);
                }
            }
        }
    }
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }

    // 3. Determine which theme tokens the generated CSS references.
    let mut referenced: BTreeSet<String> = BTreeSet::new();
    for rules in groups.values() {
        for (_, _, css) in rules {
            collect_theme_vars_str(css, &theme, &mut referenced);
        }
    }
    collect_theme_vars_str(&user.base_layer, &theme, &mut referenced);
    collect_theme_vars_str(&user.postlude, &theme, &mut referenced);
    // The preflight always relies on the default font-family tokens.
    for always in ["--font-sans", "--font-mono", "--default-font-family", "--default-mono-font-family"] {
        if theme.contains(always) {
            referenced.insert(always.to_string());
        }
    }

    // 4. Assemble the stylesheet, layer by layer, matching Tailwind v4 order.
    let mut out = String::new();
    out.push_str(VERSION_BANNER);
    out.push('\n');

    if !tw_props.is_empty() {
        out.push_str("@layer properties{");
        out.push_str(PROPERTIES_SUPPORTS);
        out.push_str("{*,:before,:after,::backdrop{");
        out.push_str(
            &tw_props
                .iter()
                .map(|prop| prop.layer_declaration())
                .collect::<Vec<_>>()
                .join(";"),
        );
        out.push_str("}}}");
    }

    out.push_str("@layer theme{:root,:host{");
    out.push_str(&theme.render(&referenced));
    out.push_str("}}");

    out.push_str("@layer base{");
    out.push_str(PREFLIGHT_CSS);
    out.push_str(&user.base_layer);
    out.push('}');

    out.push_str("@layer components;");

    out.push_str("@layer utilities{");
    for ((_, media_key), mut rules) in groups {
        rules.sort();
        let conditions: Vec<&str> = media_key.split('|').filter(|c| !c.is_empty()).collect();
        for condition in &conditions {
            out.push_str("@media (");
            out.push_str(condition);
            out.push_str("){");
        }
        for (_, _, css) in &rules {
            out.push_str(css);
        }
        for _ in &conditions {
            out.push('}');
        }
    }
    out.push('}');

    // The app's own plain (unlayered) rules follow the layers, exactly where
    // they sit relative to the `@import 'tailwindcss'` in the source.
    out.push_str(&user.postlude);

    for prop in &tw_props {
        out.push_str(&prop.property_declaration());
    }

    Ok(out)
}

/// `group`/`peer` (optionally named, `group/name`) are variant marker classes:
/// Tailwind generates no CSS for them.
fn is_marker_class(class: &str) -> bool {
    class == "group"
        || class == "peer"
        || class.starts_with("group/")
        || class.starts_with("peer/")
}

// ---------------------------------------------------------------------------
// Candidate scanning
// ---------------------------------------------------------------------------

/// Scans a JavaScript/TypeScript/JSX source for utility class candidates.
///
/// Extracts the string values that flow into `className`/`class` attributes or
/// object properties: string literals, template literals (interpolations are
/// token boundaries), ternary branches, `+` concatenations, and — transitively —
/// `const`/`let`/`var` string bindings referenced from those positions (a common
/// pattern: `const buttonBase = "inline-flex …"; <button className={buttonBase}>`).
/// Only initializers that are themselves string-shaped (literal, template, or a
/// ternary over string shapes) contribute, so arbitrary program strings (e.g. a
/// `mode === "split"` comparison) never leak in as candidates.
pub fn scan_class_candidates(source: &str, out: &mut BTreeSet<String>) {
    let mut idents: BTreeSet<String> = BTreeSet::new();
    scan_class_positions(source, out, &mut idents);
    resolve_identifier_bindings(source, out, idents);
}

/// Walks `class`/`className` attribute positions and collects tokens and
/// referenced identifiers from their value expressions.
fn scan_class_positions(source: &str, out: &mut BTreeSet<String>, idents: &mut BTreeSet<String>) {
    let bytes = source.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let Some(rel) = source[i..].find("class") else {
            break;
        };
        let start = i + rel;
        // Must be a word boundary before `class`.
        if start > 0 && is_ident_byte(bytes[start - 1]) {
            i = start + 5;
            continue;
        }
        let mut j = start + 5;
        // Optional `Name` suffix.
        if source[j..].starts_with("Name") {
            j += 4;
        }
        // The identifier must end here (not `classList`, `classNames`, ...).
        if j < bytes.len() && is_ident_byte(bytes[j]) {
            i = start + 5;
            continue;
        }
        // Skip whitespace, then require `=` or `:`.
        let mut k = skip_ws(bytes, j);
        if k >= bytes.len() || (bytes[k] != b'=' && bytes[k] != b':') {
            i = start + 5;
            continue;
        }
        k += 1;
        k = skip_ws(bytes, k);
        if k >= bytes.len() {
            break;
        }
        match bytes[k] {
            b'"' | b'\'' | b'`' => {
                i = collect_string_shape(source, k, out, idents).unwrap_or(k + 1);
            }
            b'{' => {
                // JSX expression container: read the balanced braces and treat
                // the inner expression as a class expression.
                let Some(end) = find_balanced(source, k, b'{', b'}') else {
                    break;
                };
                collect_class_expression(&source[k + 1..end], out, idents);
                i = end + 1;
            }
            _ => {
                i = start + 5;
            }
        }
    }
}

/// Collects candidates from a class-valued expression: a string/template
/// literal, an identifier reference, a ternary over class expressions (the
/// condition is ignored), a parenthesized expression, or a `+` concatenation of
/// these. Anything else contributes nothing (it is outside the precision scope
/// of the scanner, e.g. a function call).
fn collect_class_expression(expr: &str, out: &mut BTreeSet<String>, idents: &mut BTreeSet<String>) {
    let expr = expr.trim();
    if expr.is_empty() {
        return;
    }
    if let Some((then_part, else_part)) = split_ternary(expr) {
        collect_class_expression(then_part, out, idents);
        collect_class_expression(else_part, out, idents);
        return;
    }
    let bytes = expr.as_bytes();
    let mut i = 0;
    loop {
        i = skip_ws(bytes, i);
        if i >= bytes.len() {
            return;
        }
        match bytes[i] {
            b'"' | b'\'' | b'`' => {
                let Some(end) = collect_string_shape(expr, i, out, idents) else {
                    return;
                };
                i = end;
            }
            b'(' => {
                let Some(end) = find_balanced(expr, i, b'(', b')') else {
                    return;
                };
                collect_class_expression(&expr[i + 1..end], out, idents);
                i = end + 1;
            }
            b if is_ident_byte(b) && !b.is_ascii_digit() => {
                let mut j = i;
                while j < bytes.len() && is_ident_byte(bytes[j]) {
                    j += 1;
                }
                let after = skip_ws(bytes, j);
                // A plain identifier reference; calls and member accesses are
                // out of scope.
                if after >= bytes.len() || bytes[after] == b'+' {
                    idents.insert(expr[i..j].to_string());
                }
                i = j;
            }
            _ => return,
        }
        i = skip_ws(bytes, i);
        if i < bytes.len() && bytes[i] == b'+' {
            i += 1;
        } else {
            return;
        }
    }
}

/// Reads a string or template literal starting at the quote byte; tokenizes its
/// literal content into `out` and recurses into template interpolations.
/// Returns the index just past the closing quote.
fn collect_string_shape(
    source: &str,
    start: usize,
    out: &mut BTreeSet<String>,
    idents: &mut BTreeSet<String>,
) -> Option<usize> {
    let bytes = source.as_bytes();
    let quote = bytes[start];
    let mut segment = String::new();
    let mut p = start + 1;
    while p < bytes.len() {
        let c = bytes[p];
        if c == b'\\' {
            if p + 1 < bytes.len() {
                segment.push(bytes[p + 1] as char);
            }
            p += 2;
            continue;
        }
        if quote == b'`' && c == b'$' && p + 1 < bytes.len() && bytes[p + 1] == b'{' {
            tokenize_class_segment(&segment, out);
            segment.clear();
            let end = find_balanced(source, p + 1, b'{', b'}')?;
            collect_class_expression(&source[p + 2..end], out, idents);
            p = end + 1;
            continue;
        }
        if c == quote {
            tokenize_class_segment(&segment, out);
            return Some(p + 1);
        }
        segment.push(c as char);
        p += 1;
    }
    tokenize_class_segment(&segment, out);
    None
}

fn tokenize_class_segment(segment: &str, out: &mut BTreeSet<String>) {
    for token in segment.split_whitespace() {
        if !token.is_empty() && !token.contains('$') {
            out.insert(token.to_string());
        }
    }
}

/// Resolves identifiers referenced from class positions against `const`/`let`/
/// `var` bindings in the same source, transitively. Only string-shaped
/// initializers contribute candidates.
fn resolve_identifier_bindings(source: &str, out: &mut BTreeSet<String>, pending: BTreeSet<String>) {
    let mut visited: BTreeSet<String> = BTreeSet::new();
    let mut worklist: Vec<String> = pending.into_iter().collect();
    while let Some(name) = worklist.pop() {
        if !visited.insert(name.clone()) {
            continue;
        }
        for init in find_binding_initializers(source, &name) {
            let init = init.trim();
            let eligible = init.starts_with(['"', '\'', '`', '('])
                || split_ternary(init).is_some();
            if !eligible {
                continue;
            }
            let mut new_idents = BTreeSet::new();
            collect_class_expression(init, out, &mut new_idents);
            for ident in new_idents {
                if !visited.contains(&ident) {
                    worklist.push(ident);
                }
            }
        }
    }
}

/// Finds every `const|let|var <name> = <initializer>` in the source and returns
/// the initializer texts (up to the top-level `;`).
fn find_binding_initializers<'a>(source: &'a str, name: &str) -> Vec<&'a str> {
    let bytes = source.as_bytes();
    let mut found = Vec::new();
    let mut i = 0;
    while let Some(rel) = source[i..].find(name) {
        let start = i + rel;
        let end = start + name.len();
        i = end;
        // Word boundaries around the identifier.
        if (start > 0 && is_ident_byte(bytes[start - 1]))
            || (end < bytes.len() && is_ident_byte(bytes[end]))
        {
            continue;
        }
        // Preceded (over whitespace) by a declaration keyword.
        let before = source[..start].trim_end();
        let declared = ["const", "let", "var"].iter().any(|kw| {
            before.ends_with(kw)
                && before[..before.len() - kw.len()]
                    .chars()
                    .next_back()
                    .is_none_or(|c| !(c.is_ascii_alphanumeric() || c == '_' || c == '$'))
        });
        if !declared {
            continue;
        }
        // Followed (over whitespace) by a single `=`.
        let mut k = skip_ws(bytes, end);
        if k >= bytes.len() || bytes[k] != b'=' {
            continue;
        }
        k += 1;
        if k < bytes.len() && (bytes[k] == b'=' || bytes[k] == b'>') {
            continue; // `==` / `=>`
        }
        // Initializer runs to the top-level `;` (string- and bracket-aware).
        let init_start = k;
        let mut depth = 0i32;
        let mut string: Option<u8> = None;
        while k < bytes.len() {
            let c = bytes[k];
            if let Some(q) = string {
                if c == b'\\' {
                    k += 2;
                    continue;
                }
                if c == q {
                    string = None;
                }
            } else {
                match c {
                    b'"' | b'\'' | b'`' => string = Some(c),
                    b'(' | b'[' | b'{' => depth += 1,
                    b')' | b']' | b'}' => depth -= 1,
                    b';' if depth == 0 => break,
                    _ => {}
                }
            }
            k += 1;
        }
        found.push(&source[init_start..k]);
        i = k;
    }
    found
}

/// Splits `cond ? then : else` at the top level (string- and bracket-aware,
/// skipping `?.` and `??`). Returns the two branches; the condition is dropped.
fn split_ternary(expr: &str) -> Option<(&str, &str)> {
    let bytes = expr.as_bytes();
    let mut depth = 0i32;
    let mut string: Option<u8> = None;
    let mut question: Option<usize> = None;
    let mut nested = 0i32;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if let Some(q) = string {
            if c == b'\\' {
                i += 2;
                continue;
            }
            if c == q {
                string = None;
            }
            i += 1;
            continue;
        }
        match c {
            b'"' | b'\'' | b'`' => string = Some(c),
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b'?' if depth == 0 => {
                let next = bytes.get(i + 1).copied();
                if next == Some(b'.') || next == Some(b'?') {
                    i += 2;
                    continue;
                }
                if question.is_none() {
                    question = Some(i);
                } else {
                    nested += 1;
                }
            }
            b':' if depth == 0 && question.is_some() => {
                if nested == 0 {
                    let q = question.unwrap();
                    return Some((&expr[q + 1..i], &expr[i + 1..]));
                }
                nested -= 1;
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Finds the index of the closing bracket matching the opener at `open`
/// (string-aware).
fn find_balanced(source: &str, open: usize, open_byte: u8, close_byte: u8) -> Option<usize> {
    let bytes = source.as_bytes();
    debug_assert_eq!(bytes[open], open_byte);
    let mut depth = 0i32;
    let mut string: Option<u8> = None;
    let mut i = open;
    while i < bytes.len() {
        let c = bytes[i];
        if let Some(q) = string {
            if c == b'\\' {
                i += 2;
                continue;
            }
            if c == q {
                string = None;
            }
        } else if c == open_byte {
            depth += 1;
        } else if c == close_byte {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        } else if c == b'"' || c == b'\'' || c == b'`' {
            string = Some(c);
        }
        i += 1;
    }
    None
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'$'
}

fn skip_ws(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    i
}

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

/// The parsed Tailwind default theme: variable name -> value, plus source order.
struct Theme {
    values: BTreeMap<String, String>,
    order: Vec<String>,
}

impl Theme {
    fn parse(css: &str) -> Self {
        let mut values = BTreeMap::new();
        let mut order = Vec::new();
        let bytes = css.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            // Find the next custom-property declaration `--name:`.
            let Some(rel) = css[i..].find("--") else {
                break;
            };
            let name_start = i + rel;
            let mut j = name_start + 2;
            while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'-') {
                j += 1;
            }
            // Require a `:` right after the name (allowing whitespace).
            let mut k = j;
            while k < bytes.len() && bytes[k].is_ascii_whitespace() {
                k += 1;
            }
            if k >= bytes.len() || bytes[k] != b':' {
                i = j.max(name_start + 2);
                continue;
            }
            let name = css[name_start..j].to_string();
            // Value runs until the terminating `;` (values here contain no `;`).
            let value_start = k + 1;
            let Some(semi_rel) = css[value_start..].find(';') else {
                break;
            };
            let raw = &css[value_start..value_start + semi_rel];
            let value = normalize_theme_value(raw);
            if !values.contains_key(&name) {
                order.push(name.clone());
            }
            values.insert(name, value);
            i = value_start + semi_rel + 1;
        }
        Self { values, order }
    }

    fn contains(&self, name: &str) -> bool {
        self.values.contains_key(name)
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.values.get(name).map(|v| v.as_str())
    }

    /// Renders the `:root,:host` body with only the referenced tokens, in the
    /// theme's own source order (matching Tailwind's tree-shaken theme layer).
    fn render(&self, referenced: &BTreeSet<String>) -> String {
        let mut parts = Vec::new();
        for name in &self.order {
            if referenced.contains(name) {
                parts.push(format!("{name}:{}", self.values[name]));
            }
        }
        parts.join(";")
    }
}

/// Collapses whitespace and rewrites `--theme(--x, initial)` (used by the default
/// font tokens) into `var(--x)`, matching the compiled theme layer.
fn normalize_theme_value(raw: &str) -> String {
    let collapsed = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    if let Some(rest) = collapsed.strip_prefix("--theme(") {
        let inner = rest.trim_end_matches(')');
        let first = inner.split(',').next().unwrap_or("").trim();
        return format!("var({first})");
    }
    collapsed
}

/// Finds every `var(--name)` reference in the generated CSS whose `--name` is a
/// theme token, so the theme layer emits exactly the referenced tokens.
fn collect_theme_vars_str(chunk: &str, theme: &Theme, out: &mut BTreeSet<String>) {
    let mut rest = chunk;
    while let Some(pos) = rest.find("var(--") {
        let after = &rest[pos + 4..];
        let end = after
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '-'))
            .unwrap_or(after.len());
        let name = &after[..end];
        if theme.contains(name) {
            out.insert(name.to_string());
        }
        rest = &after[end..];
    }
}

// ---------------------------------------------------------------------------
// Registered `--tw-*` custom properties
// ---------------------------------------------------------------------------

/// A registered `--tw-*` custom property a utility depends on. Variant order is
/// Tailwind's canonical registration order (the derived `Ord` follows it).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum TwProp {
    TranslateX,
    TranslateY,
    TranslateZ,
    SpaceYReverse,
    BorderStyle,
    Leading,
    FontWeight,
    Tracking,
    Ordinal,
    SlashedZero,
    NumericFigure,
    NumericSpacing,
    NumericFraction,
    Shadow,
    ShadowColor,
    ShadowAlpha,
    InsetShadow,
    InsetShadowColor,
    InsetShadowAlpha,
    RingColor,
    RingShadow,
    InsetRingColor,
    InsetRingShadow,
    RingInset,
    RingOffsetWidth,
    RingOffsetColor,
    RingOffsetShadow,
    OutlineStyle,
    Content,
}

impl TwProp {
    /// `name, layer initial value, @property syntax, @property initial value`.
    fn spec(self) -> (&'static str, &'static str, &'static str, Option<&'static str>) {
        match self {
            TwProp::TranslateX => ("--tw-translate-x", "0", "\"*\"", Some("0")),
            TwProp::TranslateY => ("--tw-translate-y", "0", "\"*\"", Some("0")),
            TwProp::TranslateZ => ("--tw-translate-z", "0", "\"*\"", Some("0")),
            TwProp::SpaceYReverse => ("--tw-space-y-reverse", "0", "\"*\"", Some("0")),
            TwProp::BorderStyle => ("--tw-border-style", "solid", "\"*\"", Some("solid")),
            TwProp::Leading => ("--tw-leading", "initial", "\"*\"", None),
            TwProp::FontWeight => ("--tw-font-weight", "initial", "\"*\"", None),
            TwProp::Tracking => ("--tw-tracking", "initial", "\"*\"", None),
            TwProp::Ordinal => ("--tw-ordinal", "initial", "\"*\"", None),
            TwProp::SlashedZero => ("--tw-slashed-zero", "initial", "\"*\"", None),
            TwProp::NumericFigure => ("--tw-numeric-figure", "initial", "\"*\"", None),
            TwProp::NumericSpacing => ("--tw-numeric-spacing", "initial", "\"*\"", None),
            TwProp::NumericFraction => ("--tw-numeric-fraction", "initial", "\"*\"", None),
            TwProp::Shadow => ("--tw-shadow", "0 0 #0000", "\"*\"", Some("0 0 #0000")),
            TwProp::ShadowColor => ("--tw-shadow-color", "initial", "\"*\"", None),
            TwProp::ShadowAlpha => ("--tw-shadow-alpha", "100%", "\"<percentage>\"", Some("100%")),
            TwProp::InsetShadow => ("--tw-inset-shadow", "0 0 #0000", "\"*\"", Some("0 0 #0000")),
            TwProp::InsetShadowColor => ("--tw-inset-shadow-color", "initial", "\"*\"", None),
            TwProp::InsetShadowAlpha => {
                ("--tw-inset-shadow-alpha", "100%", "\"<percentage>\"", Some("100%"))
            }
            TwProp::RingColor => ("--tw-ring-color", "initial", "\"*\"", None),
            TwProp::RingShadow => ("--tw-ring-shadow", "0 0 #0000", "\"*\"", Some("0 0 #0000")),
            TwProp::InsetRingColor => ("--tw-inset-ring-color", "initial", "\"*\"", None),
            TwProp::InsetRingShadow => {
                ("--tw-inset-ring-shadow", "0 0 #0000", "\"*\"", Some("0 0 #0000"))
            }
            TwProp::RingInset => ("--tw-ring-inset", "initial", "\"*\"", None),
            TwProp::RingOffsetWidth => {
                ("--tw-ring-offset-width", "0px", "\"<length>\"", Some("0"))
            }
            TwProp::RingOffsetColor => ("--tw-ring-offset-color", "#fff", "\"*\"", Some("#fff")),
            TwProp::RingOffsetShadow => {
                ("--tw-ring-offset-shadow", "0 0 #0000", "\"*\"", Some("0 0 #0000"))
            }
            TwProp::OutlineStyle => ("--tw-outline-style", "solid", "\"*\"", Some("solid")),
            TwProp::Content => ("--tw-content", "\"\"", "\"*\"", Some("\"\"")),
        }
    }

    fn layer_declaration(self) -> String {
        let (name, initial, _, _) = self.spec();
        format!("{name}:{initial}")
    }

    fn property_declaration(self) -> String {
        let (name, _, syntax, initial) = self.spec();
        match initial {
            Some(value) => format!(
                "@property {name}{{syntax:{syntax};inherits:false;initial-value:{value}}}"
            ),
            None => format!("@property {name}{{syntax:{syntax};inherits:false}}"),
        }
    }
}

/// Registers the full box-shadow property group (Tailwind registers the whole
/// group whenever any shadow/ring utility appears, because they compose into one
/// `box-shadow`).
fn register_shadow_group(tw_props: &mut BTreeSet<TwProp>) {
    for prop in [
        TwProp::Shadow,
        TwProp::ShadowColor,
        TwProp::ShadowAlpha,
        TwProp::InsetShadow,
        TwProp::InsetShadowColor,
        TwProp::InsetShadowAlpha,
        TwProp::RingColor,
        TwProp::RingShadow,
        TwProp::InsetRingColor,
        TwProp::InsetRingShadow,
        TwProp::RingInset,
        TwProp::RingOffsetWidth,
        TwProp::RingOffsetColor,
        TwProp::RingOffsetShadow,
    ] {
        tw_props.insert(prop);
    }
}

// ---------------------------------------------------------------------------
// Variants
// ---------------------------------------------------------------------------

/// The parsed variant chain of a class: pseudo suffix, media wrappers, output
/// order, and whether the rule targets a `::before`/`::after` pseudo-element
/// (which injects `content:var(--tw-content)`).
struct VariantSpec {
    pseudo: String,
    /// `:is(...)` clause inserted right after the class selector (group-hover).
    is_clause: String,
    media: Vec<String>,
    order: u8,
    inject_content: bool,
}

/// Output order indices per variant, mirroring the reference stylesheet:
/// base < before/after < hover < focus < focus-visible < active < disabled
/// < breakpoints < dark.
fn parse_variants(
    segments: &[&str],
    class: &str,
    theme: &Theme,
    tw_props: &mut BTreeSet<TwProp>,
    custom_variants: &std::collections::BTreeMap<String, String>,
) -> Result<VariantSpec, String> {
    let mut spec = VariantSpec {
        pseudo: String::new(),
        is_clause: String::new(),
        media: Vec::new(),
        order: 0,
        inject_content: false,
    };
    let push_media = |media: &mut Vec<String>, condition: &str| {
        if !media.iter().any(|m| m == condition) {
            media.push(condition.to_string());
        }
    };
    for variant in segments {
        match *variant {
            "hover" => {
                spec.pseudo.push_str(":hover");
                push_media(&mut spec.media, "hover:hover");
                spec.order = spec.order.max(2);
            }
            "group-hover" => {
                spec.is_clause = ":is(:where(.group):hover *)".to_string();
                push_media(&mut spec.media, "hover:hover");
                spec.order = spec.order.max(2);
            }
            "focus" => {
                spec.pseudo.push_str(":focus");
                spec.order = spec.order.max(3);
            }
            "focus-visible" => {
                spec.pseudo.push_str(":focus-visible");
                spec.order = spec.order.max(4);
            }
            "active" => {
                spec.pseudo.push_str(":active");
                spec.order = spec.order.max(3);
            }
            "disabled" => {
                spec.pseudo.push_str(":disabled");
                spec.order = spec.order.max(5);
            }
            "before" | "after" => {
                spec.pseudo.push(':');
                spec.pseudo.push_str(variant);
                spec.inject_content = true;
                tw_props.insert(TwProp::Content);
                spec.order = spec.order.max(1);
            }
            "backdrop" => {
                spec.pseudo.push_str("::backdrop");
            }
            name if custom_variants.contains_key(name) => {
                // A `@custom-variant` overrides any built-in meaning: the
                // `&`-rooted template appends to the candidate selector
                // (`dark:x` -> `.dark\:x:where(.dark, .dark *)`).
                let template = &custom_variants[name];
                spec.pseudo.push_str(&template[1..]);
                spec.order = spec.order.max(12);
            }
            "dark" => {
                push_media(&mut spec.media, "prefers-color-scheme:dark");
                spec.order = spec.order.max(12);
            }
            bp @ ("sm" | "md" | "lg" | "xl" | "2xl") => {
                let var = format!("--breakpoint-{bp}");
                let value = theme.get(&var).ok_or_else(|| {
                    format!("unknown breakpoint `{bp}:` in class `{class}`")
                })?;
                push_media(&mut spec.media, &format!("width>={value}"));
                let index = ["sm", "md", "lg", "xl", "2xl"]
                    .iter()
                    .position(|b| *b == bp)
                    .unwrap() as u8;
                spec.order = spec.order.max(7 + index);
            }
            other => {
                return Err(format!(
                    "unsupported Tailwind variant `{other}:` in class `{class}`: the native compiler does not yet generate it. Extend src/tailwind.rs (do not silently drop it)."
                ));
            }
        }
    }
    Ok(spec)
}

struct RenderedRule {
    css: String,
    order: u8,
    media_key: String,
    rank: u16,
}

/// Whether a candidate's base (variants stripped) starts with a prefix any
/// utility family owns, or is one of the exact utility keywords. Only these
/// candidates hard-error when unsupported; everything else is a non-utility
/// source token Tailwind's scanner would also ignore.
fn utility_root_recognized(class: &str) -> bool {
    let segments = split_variants(class);
    let base = segments.last().copied().unwrap_or(class);
    let base = base.strip_prefix('-').unwrap_or(base);
    const PREFIXES: &[&str] = &[
        "bg-", "text-", "font-", "border-", "rounded-", "ring-", "outline-", "shadow-",
        "inset-", "top-", "right-", "bottom-", "left-", "z-", "translate-", "rotate-",
        "scale-", "skew-", "w-", "h-", "min-w-", "min-h-", "max-w-", "max-h-", "size-",
        "m-", "mx-", "my-", "mt-", "mr-", "mb-", "ml-", "p-", "px-", "py-", "pt-", "pr-",
        "pb-", "pl-", "gap-", "space-", "flex-", "grid-", "col-", "row-", "justify-",
        "items-", "content-", "self-", "place-", "order-", "leading-", "tracking-",
        "align-", "list-", "decoration-", "underline-", "overflow-", "overscroll-",
        "cursor-", "select-", "pointer-events-", "opacity-", "transition-", "duration-",
        "delay-", "ease-", "animate-", "fill-", "stroke-", "object-", "aspect-",
        "columns-", "break-", "whitespace-", "line-clamp-", "backdrop-", "blur-",
        "brightness-", "contrast-", "divide-", "accent-", "caret-", "scroll-",
        "snap-", "touch-", "will-change-", "from-", "via-", "to-",
    ];
    const EXACT: &[&str] = &[
        "flex", "grid", "block", "inline", "inline-block", "inline-flex", "inline-grid",
        "hidden", "contents", "static", "fixed", "absolute", "relative", "sticky",
        "isolate", "visible", "invisible", "container", "italic", "not-italic",
        "underline", "overline", "line-through", "no-underline", "uppercase",
        "lowercase", "capitalize", "normal-case", "truncate", "antialiased",
        "subpixel-antialiased", "rounded", "rounded-full", "border", "ring",
        "ring-inset", "shadow", "outline", "outline-none", "transition", "grow",
        "shrink", "tabular-nums", "sr-only", "not-sr-only", "group", "peer",
    ];
    PREFIXES.iter().any(|prefix| base.starts_with(prefix))
        || EXACT.contains(&base)
}

/// Renders a full utility class (with any variant chain) into a CSS rule string
/// plus its output grouping. Errors if the token is not a recognized utility.
fn render_utility(
    class: &str,
    theme: &Theme,
    tw_props: &mut BTreeSet<TwProp>,
    custom_variants: &std::collections::BTreeMap<String, String>,
) -> Result<RenderedRule, String> {
    // Split on `:` only outside brackets: arbitrary values may contain `:`
    // (e.g. `bg-[color:var(--x)]`).
    let mut segments = split_variants(class);
    let base = segments
        .pop()
        .ok_or_else(|| format!("empty class candidate `{class}`"))?;
    let spec = parse_variants(&segments, class, theme, tw_props, custom_variants)?;

    let utility = generate_utility(base, class, theme, tw_props)?;
    let escaped = escape_class(class);
    let selector = match utility.selector {
        SelectorKind::Class => format!(".{escaped}{}{}", spec.is_clause, spec.pseudo),
        SelectorKind::SpaceChildren => format!(":where(.{escaped}>:not(:last-child))"),
    };
    let mut decls = utility.decls;
    if spec.inject_content && !decls.iter().any(|(prop, _)| prop == "content") {
        decls.insert(0, ("content".to_string(), "var(--tw-content)".to_string()));
    }
    let body = decls
        .iter()
        .map(|(prop, value)| format!("{prop}:{value}"))
        .collect::<Vec<_>>()
        .join(";");
    Ok(RenderedRule {
        css: format!("{selector}{{{body}}}"),
        order: spec.order,
        media_key: spec.media.join("|"),
        rank: utility.rank,
    })
}

/// Splits a class into `:`-separated variant segments, ignoring `:` inside
/// `[…]` arbitrary values.
fn split_variants(class: &str) -> Vec<&str> {
    let bytes = class.as_bytes();
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'[' => depth += 1,
            b']' => depth -= 1,
            b':' if depth == 0 => {
                parts.push(&class[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&class[start..]);
    parts
}

enum SelectorKind {
    Class,
    SpaceChildren,
}

struct Utility {
    selector: SelectorKind,
    decls: Vec<(String, String)>,
    /// Output rank within a group: orders overlapping shorthand families
    /// (`p` < `px` < `py` < `pt` …) the way Tailwind does; unrelated utilities
    /// share a default rank and sort by name.
    rank: u16,
}

impl Utility {
    fn simple(decls: Vec<(&str, String)>) -> Utility {
        Utility::ranked(decls, 100)
    }

    fn ranked(decls: Vec<(&str, String)>, rank: u16) -> Utility {
        Utility {
            selector: SelectorKind::Class,
            decls: decls
                .into_iter()
                .map(|(prop, value)| (prop.to_string(), value))
                .collect(),
            rank,
        }
    }
}

// ---------------------------------------------------------------------------
// Utility generation
// ---------------------------------------------------------------------------

/// The general utility generator. `base` is the class with variants stripped;
/// `full` is the original token (for error messages). Returns a hard error naming
/// the token if it matches a known utility family but references an unknown value,
/// or if the family itself is unimplemented.
fn generate_utility(
    base: &str,
    full: &str,
    theme: &Theme,
    tw_props: &mut BTreeSet<TwProp>,
) -> Result<Utility, String> {
    if let Some(decls) = keyword_utility(base) {
        return Ok(Utility::simple(decls));
    }

    // sr-only: the screen-reader-only composite.
    if base == "sr-only" {
        return Ok(Utility::simple(vec![
            ("position", "absolute".to_string()),
            ("width", "1px".to_string()),
            ("height", "1px".to_string()),
            ("padding", "0".to_string()),
            ("margin", "-1px".to_string()),
            ("overflow", "hidden".to_string()),
            ("clip-path", "inset(50%)".to_string()),
            ("white-space", "nowrap".to_string()),
            ("border-width", "0".to_string()),
        ]));
    }

    if base == "truncate" {
        return Ok(Utility::simple(vec![
            ("overflow", "hidden".to_string()),
            ("text-overflow", "ellipsis".to_string()),
            ("white-space", "nowrap".to_string()),
        ]));
    }

    // Negative-capable families share the leading `-` strip.
    let (negative, positive_base) = match base.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, base),
    };

    // Translate: translate-x-/translate-y- over fractions, spacing, px, full.
    if let Some(rest) = positive_base.strip_prefix("translate-") {
        if let Some((axis @ ("x" | "y"), value)) = rest.split_once('-') {
            let resolved = translate_value(value, negative).ok_or_else(|| unknown(full))?;
            for prop in [TwProp::TranslateX, TwProp::TranslateY, TwProp::TranslateZ] {
                tw_props.insert(prop);
            }
            let var = format!("--tw-translate-{axis}");
            return Ok(Utility {
                selector: SelectorKind::Class,
                decls: vec![
                    (var, resolved),
                    (
                        "translate".to_string(),
                        "var(--tw-translate-x) var(--tw-translate-y)".to_string(),
                    ),
                ],
                rank: 100,
            });
        }
        return Err(unknown(full));
    }

    // Position offsets: inset/inset-x/inset-y/top/right/bottom/left.
    // Longer prefixes first so `inset-y-0` is not consumed by `inset`.
    let position_families: [(&str, &str, u16); 7] = [
        ("inset-x", "inset-inline", 31),
        ("inset-y", "inset-block", 32),
        ("inset", "inset", 30),
        ("top", "top", 33),
        ("right", "right", 34),
        ("bottom", "bottom", 35),
        ("left", "left", 36),
    ];
    for (prefix, property, rank) in position_families {
        if let Some(value) = strip_family(positive_base, prefix) {
            let resolved = offset_value(value, negative).ok_or_else(|| unknown(full))?;
            return Ok(Utility::ranked(vec![(property, resolved)], rank));
        }
    }

    // z-index.
    if let Some(value) = strip_family(positive_base, "z") {
        if value.bytes().all(|b| b.is_ascii_digit()) && !value.is_empty() {
            let z = if negative { format!("-{value}") } else { value.to_string() };
            return Ok(Utility::simple(vec![("z-index", z)]));
        }
        return Err(unknown(full));
    }

    if negative {
        // Negative margins fall through below; other negatives are unknown.
        if !(positive_base.starts_with('m')) {
            return Err(unknown(full));
        }
    }

    // Sizing: w/h/min-w/min-h/max-w/max-h/size.
    if let Some(utility) = sizing_utility(positive_base, full, theme)? {
        if negative {
            return Err(unknown(full));
        }
        return Ok(utility);
    }

    // `border…`: side widths, side colors, colors, or the bare side shorthand.
    if base == "border" || base.starts_with("border-") {
        return border_utility(base, full, theme, tw_props);
    }

    // gap-x / gap-y (before the plain `gap-` family consumes the prefix).
    if let Some(n) = base.strip_prefix("gap-x-") {
        let value = spacing_value(n, false).ok_or_else(|| unknown(full))?;
        return Ok(Utility::simple(vec![("column-gap", value)]));
    }
    if let Some(n) = base.strip_prefix("gap-y-") {
        let value = spacing_value(n, false).ok_or_else(|| unknown(full))?;
        return Ok(Utility::simple(vec![("row-gap", value)]));
    }

    // Spacing: padding/margin/gap families (margins support `auto` and `-`).
    if let Some(utility) = spacing_utility(positive_base, full, negative)? {
        return Ok(utility);
    }

    // space-y-<n>: adjacent-sibling margin utility.
    if let Some(n) = base.strip_prefix("space-y-") {
        let value = spacing_value(n, false).ok_or_else(|| unknown(full))?;
        tw_props.insert(TwProp::SpaceYReverse);
        return Ok(Utility {
            selector: SelectorKind::SpaceChildren,
            decls: vec![
                ("--tw-space-y-reverse".to_string(), "0".to_string()),
                (
                    "margin-block-start".to_string(),
                    format!("calc({value} * var(--tw-space-y-reverse))"),
                ),
                (
                    "margin-block-end".to_string(),
                    format!("calc({value} * calc(1 - var(--tw-space-y-reverse)))"),
                ),
            ],
            rank: 100,
        });
    }

    // grid-template-columns/rows.
    if let Some(value) = base.strip_prefix("grid-cols-") {
        let resolved = grid_template_value(value).ok_or_else(|| unknown(full))?;
        return Ok(Utility::simple(vec![("grid-template-columns", resolved)]));
    }
    if let Some(value) = base.strip_prefix("grid-rows-") {
        let resolved = grid_template_value(value).ok_or_else(|| unknown(full))?;
        return Ok(Utility::simple(vec![("grid-template-rows", resolved)]));
    }

    // flex-shrink / flex-grow.
    if base == "shrink" {
        return Ok(Utility::simple(vec![("flex-shrink", "1".to_string())]));
    }
    if let Some(n) = base.strip_prefix("shrink-") {
        if n.bytes().all(|b| b.is_ascii_digit()) && !n.is_empty() {
            return Ok(Utility::simple(vec![("flex-shrink", n.to_string())]));
        }
        return Err(unknown(full));
    }
    if base == "grow" {
        return Ok(Utility::simple(vec![("flex-grow", "1".to_string())]));
    }
    if let Some(n) = base.strip_prefix("grow-") {
        if n.bytes().all(|b| b.is_ascii_digit()) && !n.is_empty() {
            return Ok(Utility::simple(vec![("flex-grow", n.to_string())]));
        }
        return Err(unknown(full));
    }

    // rounded-<size>: border-radius from the theme radius scale.
    if base == "rounded" {
        return Ok(Utility::ranked(vec![("border-radius", "0.25rem".to_string())], 45));
    }
    if let Some(size) = base.strip_prefix("rounded-") {
        if size == "full" {
            return Ok(Utility::ranked(
                vec![("border-radius", "3.40282e38px".to_string())],
                45,
            ));
        }
        if size == "none" {
            return Ok(Utility::ranked(vec![("border-radius", "0".to_string())], 45));
        }
        let var = format!("--radius-{size}");
        if theme.contains(&var) {
            return Ok(Utility::ranked(
                vec![("border-radius", format!("var({var})"))],
                45,
            ));
        }
        return Err(unknown(full));
    }

    // cursor-<keyword>.
    if let Some(kw) = base.strip_prefix("cursor-") {
        if is_cursor_keyword(kw) {
            return Ok(Utility::simple(vec![("cursor", kw.to_string())]));
        }
        return Err(unknown(full));
    }

    // opacity-<n>: percentage over 100.
    if let Some(n) = base.strip_prefix("opacity-") {
        if n.bytes().all(|b| b.is_ascii_digit()) && !n.is_empty() {
            let value: u32 = n.parse().map_err(|_| unknown(full))?;
            return Ok(Utility::simple(vec![("opacity", percent_fraction(value))]));
        }
        return Err(unknown(full));
    }

    // line-clamp-<n>.
    if let Some(n) = base.strip_prefix("line-clamp-") {
        if n.bytes().all(|b| b.is_ascii_digit()) && !n.is_empty() {
            return Ok(Utility::simple(vec![
                ("overflow", "hidden".to_string()),
                ("display", "-webkit-box".to_string()),
                ("-webkit-box-orient", "vertical".to_string()),
                ("-webkit-line-clamp", n.to_string()),
            ]));
        }
        return Err(unknown(full));
    }

    // underline-offset-<n>.
    if let Some(n) = base.strip_prefix("underline-offset-") {
        if n.bytes().all(|b| b.is_ascii_digit()) && !n.is_empty() {
            return Ok(Utility::simple(vec![(
                "text-underline-offset",
                format!("{n}px"),
            )]));
        }
        return Err(unknown(full));
    }

    // leading-<value>: --tw-leading + line-height.
    if let Some(rest) = base.strip_prefix("leading-") {
        let value = if rest == "none" {
            Some("1".to_string())
        } else if theme.contains(&format!("--leading-{rest}")) {
            Some(format!("var(--leading-{rest})"))
        } else {
            spacing_value(rest, false)
        };
        let value = value.ok_or_else(|| unknown(full))?;
        tw_props.insert(TwProp::Leading);
        return Ok(Utility::simple(vec![
            ("--tw-leading", value.clone()),
            ("line-height", value),
        ]));
    }

    // tracking-<name>: --tw-tracking + letter-spacing.
    if let Some(rest) = base.strip_prefix("tracking-") {
        let var = format!("--tracking-{rest}");
        if theme.contains(&var) {
            tw_props.insert(TwProp::Tracking);
            return Ok(Utility::simple(vec![
                ("--tw-tracking", format!("var({var})")),
                ("letter-spacing", format!("var({var})")),
            ]));
        }
        return Err(unknown(full));
    }

    // tabular-nums (font-variant-numeric composition).
    if base == "tabular-nums" {
        for prop in [
            TwProp::Ordinal,
            TwProp::SlashedZero,
            TwProp::NumericFigure,
            TwProp::NumericSpacing,
            TwProp::NumericFraction,
        ] {
            tw_props.insert(prop);
        }
        return Ok(Utility::simple(vec![
            ("--tw-numeric-spacing", "tabular-nums".to_string()),
            (
                "font-variant-numeric",
                "var(--tw-ordinal,) var(--tw-slashed-zero,) var(--tw-numeric-figure,) var(--tw-numeric-spacing,) var(--tw-numeric-fraction,)".to_string(),
            ),
        ]));
    }

    // shadow-<size> from the theme shadow scale.
    if let Some(size) = base.strip_prefix("shadow-") {
        if let Some(value) = theme.get(&format!("--shadow-{size}")) {
            let shadow = wrap_shadow_colors(value);
            register_shadow_group(tw_props);
            return Ok(Utility::simple(vec![
                ("--tw-shadow", shadow),
                ("box-shadow", BOX_SHADOW_CHAIN.to_string()),
            ]));
        }
        return Err(unknown(full));
    }

    // ring / ring-<n> / ring-inset / ring-offset-<n> / ring-<color>.
    if base == "ring" || base.starts_with("ring-") {
        return ring_utility(base, full, theme, tw_props);
    }

    // outline / outline-none.
    if base == "outline" {
        tw_props.insert(TwProp::OutlineStyle);
        return Ok(Utility::simple(vec![
            ("outline-style", "var(--tw-outline-style)".to_string()),
            ("outline-width", "1px".to_string()),
        ]));
    }
    if base == "outline-none" {
        tw_props.insert(TwProp::OutlineStyle);
        return Ok(Utility::simple(vec![
            ("--tw-outline-style", "none".to_string()),
            ("outline-style", "none".to_string()),
        ]));
    }

    // transition families.
    if let Some(decls) = transition_utility(base) {
        return Ok(Utility::simple(decls));
    }

    // content-[...]: --tw-content + content.
    if let Some(value) = base.strip_prefix("content-") {
        let inner = arbitrary_value(value).ok_or_else(|| unknown(full))?;
        tw_props.insert(TwProp::Content);
        return Ok(Utility::simple(vec![
            ("--tw-content", inner.clone()),
            ("content", "var(--tw-content)".to_string()),
        ]));
    }

    // accent-<color>.
    if let Some(rest) = base.strip_prefix("accent-") {
        let value = color_value(rest, theme).ok_or_else(|| unknown(full))?;
        return Ok(Utility::simple(vec![("accent-color", value)]));
    }

    // bg-<color>.
    if let Some(color) = base.strip_prefix("bg-") {
        let value = color_value(color, theme).ok_or_else(|| unknown(full))?;
        return Ok(Utility::simple(vec![("background-color", value)]));
    }

    // font-<weight> (or font-<family>).
    if let Some(rest) = base.strip_prefix("font-") {
        if is_font_weight(rest) {
            tw_props.insert(TwProp::FontWeight);
            let var = format!("--font-weight-{rest}");
            return Ok(Utility::simple(vec![
                ("--tw-font-weight", format!("var({var})")),
                ("font-weight", format!("var({var})")),
            ]));
        }
        if matches!(rest, "sans" | "serif" | "mono") {
            return Ok(Utility::simple(vec![(
                "font-family",
                format!("var(--font-{rest})"),
            )]));
        }
        return Err(unknown(full));
    }

    // text-<align>/<overflow>/<size>/<color>/[arbitrary].
    if let Some(rest) = base.strip_prefix("text-") {
        if matches!(rest, "left" | "center" | "right" | "justify" | "start" | "end") {
            return Ok(Utility::simple(vec![("text-align", rest.to_string())]));
        }
        if rest == "ellipsis" || rest == "clip" {
            return Ok(Utility::simple(vec![("text-overflow", rest.to_string())]));
        }
        if is_text_size(rest) {
            let size = format!("--text-{rest}");
            let leading = format!("--text-{rest}--line-height");
            return Ok(Utility::simple(vec![
                ("font-size", format!("var({size})")),
                (
                    "line-height",
                    format!("var(--tw-leading,var({leading}))"),
                ),
            ]));
        }
        if rest.starts_with('[') {
            // `text-[color:…]` is a color; a bare arbitrary value is a size.
            if let Some(value) = color_value(rest, theme) {
                return Ok(Utility::simple(vec![("color", value)]));
            }
            let inner = arbitrary_value(rest).ok_or_else(|| unknown(full))?;
            return Ok(Utility::simple(vec![("font-size", inner)]));
        }
        let value = color_value(rest, theme).ok_or_else(|| unknown(full))?;
        return Ok(Utility::simple(vec![("color", value)]));
    }

    Err(unknown(full))
}

/// Single-keyword utilities.
fn keyword_utility(base: &str) -> Option<Vec<(&'static str, String)>> {
    let decls: Vec<(&'static str, String)> = match base {
        "block" => vec![("display", "block".into())],
        "inline-block" => vec![("display", "inline-block".into())],
        "inline" => vec![("display", "inline".into())],
        "flex" => vec![("display", "flex".into())],
        "inline-flex" => vec![("display", "inline-flex".into())],
        "grid" => vec![("display", "grid".into())],
        "inline-grid" => vec![("display", "inline-grid".into())],
        "contents" => vec![("display", "contents".into())],
        "hidden" => vec![("display", "none".into())],
        "static" => vec![("position", "static".into())],
        "fixed" => vec![("position", "fixed".into())],
        "absolute" => vec![("position", "absolute".into())],
        "relative" => vec![("position", "relative".into())],
        "sticky" => vec![("position", "sticky".into())],
        "isolate" => vec![("isolation", "isolate".into())],
        "visible" => vec![("visibility", "visible".into())],
        "invisible" => vec![("visibility", "hidden".into())],
        "flex-col" => vec![("flex-direction", "column".into())],
        "flex-row" => vec![("flex-direction", "row".into())],
        "flex-wrap" => vec![("flex-wrap", "wrap".into())],
        "flex-nowrap" => vec![("flex-wrap", "nowrap".into())],
        "flex-1" => vec![("flex", "1".into())],
        "items-center" => vec![("align-items", "center".into())],
        "items-start" => vec![("align-items", "flex-start".into())],
        "items-end" => vec![("align-items", "flex-end".into())],
        "items-stretch" => vec![("align-items", "stretch".into())],
        "items-baseline" => vec![("align-items", "baseline".into())],
        "justify-center" => vec![("justify-content", "center".into())],
        "justify-between" => vec![("justify-content", "space-between".into())],
        "justify-start" => vec![("justify-content", "flex-start".into())],
        "justify-end" => vec![("justify-content", "flex-end".into())],
        "list-disc" => vec![("list-style-type", "disc".into())],
        "list-decimal" => vec![("list-style-type", "decimal".into())],
        "list-none" => vec![("list-style-type", "none".into())],
        "uppercase" => vec![("text-transform", "uppercase".into())],
        "lowercase" => vec![("text-transform", "lowercase".into())],
        "capitalize" => vec![("text-transform", "capitalize".into())],
        "italic" => vec![("font-style", "italic".into())],
        "not-italic" => vec![("font-style", "normal".into())],
        "underline" => vec![("text-decoration-line", "underline".into())],
        "line-through" => vec![("text-decoration-line", "line-through".into())],
        "no-underline" => vec![("text-decoration-line", "none".into())],
        "whitespace-nowrap" => vec![("white-space", "nowrap".into())],
        "whitespace-pre" => vec![("white-space", "pre".into())],
        "whitespace-pre-wrap" => vec![("white-space", "pre-wrap".into())],
        "break-all" => vec![("word-break", "break-all".into())],
        "break-words" => vec![("overflow-wrap", "break-word".into())],
        "overflow-auto" => vec![("overflow", "auto".into())],
        "overflow-hidden" => vec![("overflow", "hidden".into())],
        "overflow-visible" => vec![("overflow", "visible".into())],
        "overflow-scroll" => vec![("overflow", "scroll".into())],
        "overflow-clip" => vec![("overflow", "clip".into())],
        "overflow-x-auto" => vec![("overflow-x", "auto".into())],
        "overflow-y-auto" => vec![("overflow-y", "auto".into())],
        "overflow-x-hidden" => vec![("overflow-x", "hidden".into())],
        "overflow-y-hidden" => vec![("overflow-y", "hidden".into())],
        "select-none" => {
            vec![("-webkit-user-select", "none".into()), ("user-select", "none".into())]
        }
        "select-text" => {
            vec![("-webkit-user-select", "text".into()), ("user-select", "text".into())]
        }
        "select-all" => {
            vec![("-webkit-user-select", "all".into()), ("user-select", "all".into())]
        }
        "select-auto" => {
            vec![("-webkit-user-select", "auto".into()), ("user-select", "auto".into())]
        }
        "pointer-events-none" => vec![("pointer-events", "none".into())],
        "pointer-events-auto" => vec![("pointer-events", "auto".into())],
        _ => return None,
    };
    Some(decls)
}

fn unknown(full: &str) -> String {
    format!(
        "unsupported Tailwind utility class `{full}`: the native compiler does not yet generate it. Extend src/tailwind.rs (do not silently drop it)."
    )
}

/// Strips `<prefix>-` from a family token (`left-1/2` with prefix `left` gives
/// `1/2`).
fn strip_family<'a>(base: &'a str, prefix: &str) -> Option<&'a str> {
    base.strip_prefix(prefix)?.strip_prefix('-')
}

/// Position offset values: spacing steps, fractions, `px`, `full`, `auto`, and
/// arbitrary lengths.
fn offset_value(value: &str, negative: bool) -> Option<String> {
    if value == "auto" {
        return (!negative).then(|| "auto".to_string());
    }
    if value == "px" {
        return Some(if negative { "-1px" } else { "1px" }.to_string());
    }
    if value == "full" {
        return Some(if negative { "-100%" } else { "100%" }.to_string());
    }
    if let Some(percent) = fraction_percent(value) {
        return Some(if negative { format!("-{percent}") } else { percent });
    }
    if let Some(inner) = arbitrary_value(value) {
        return Some(if negative {
            format!("calc({inner} * -1)")
        } else {
            inner
        });
    }
    spacing_value(value, negative)
}

/// Translate values: fractions become percentage calcs (kept in Tailwind's
/// nested-calc form), numbers use the spacing scale, plus `px`/`full`.
fn translate_value(value: &str, negative: bool) -> Option<String> {
    if let Some((numerator, denominator)) = parse_fraction(value) {
        let inner = format!("calc({numerator} / {denominator} * 100%)");
        return Some(if negative {
            format!("calc({inner} * -1)")
        } else {
            inner
        });
    }
    if value == "full" {
        return Some(if negative { "-100%" } else { "100%" }.to_string());
    }
    if value == "px" {
        return Some(if negative { "-1px" } else { "1px" }.to_string());
    }
    if let Some(inner) = arbitrary_value(value) {
        return Some(if negative {
            format!("calc({inner} * -1)")
        } else {
            inner
        });
    }
    spacing_value(value, negative)
}

fn parse_fraction(value: &str) -> Option<(u32, u32)> {
    let (n, d) = value.split_once('/')?;
    let n: u32 = n.parse().ok()?;
    let d: u32 = d.parse().ok()?;
    (d != 0).then_some((n, d))
}

/// A fraction as a folded percentage (`1/2` -> `50%`), matching the reference
/// minifier's constant folding.
fn fraction_percent(value: &str) -> Option<String> {
    let (n, d) = parse_fraction(value)?;
    let percent = (n as f64) * 100.0 / (d as f64);
    Some(format!("{}%", format_number(percent)))
}

/// Formats a float without a trailing `.0` and with at most four decimals.
fn format_number(value: f64) -> String {
    if (value - value.round()).abs() < 1e-9 {
        format!("{}", value.round() as i64)
    } else {
        let s = format!("{value:.4}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// An `opacity-<n>` percentage as a fraction, minified without a leading zero
/// (`45` -> `.45`), matching the reference output.
fn percent_fraction(value: u32) -> String {
    if value.is_multiple_of(100) {
        return (value / 100).to_string();
    }
    let s = format_number(value as f64 / 100.0);
    if let Some(stripped) = s.strip_prefix("0.") {
        format!(".{stripped}")
    } else {
        s
    }
}

/// The bracketed arbitrary value of a token, with Tailwind's underscore-to-space
/// rewriting (`[auto_1fr]` -> `auto 1fr`; `\_` stays a literal underscore).
fn arbitrary_value(value: &str) -> Option<String> {
    let inner = value.strip_prefix('[')?.strip_suffix(']')?;
    if inner.is_empty() {
        return None;
    }
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' if chars.peek() == Some(&'_') => {
                out.push('_');
                chars.next();
            }
            '_' => out.push(' '),
            other => out.push(other),
        }
    }
    Some(out)
}

/// Sizing utilities (`w-`, `h-`, `min-*`, `max-*`, `size-`). Returns `Ok(None)`
/// when the prefix is not a sizing family.
fn sizing_utility(base: &str, full: &str, theme: &Theme) -> Result<Option<Utility>, String> {
    let families: [(&str, &[&str], char); 7] = [
        ("w", &["width"], 'w'),
        ("h", &["height"], 'h'),
        ("min-w", &["min-width"], 'w'),
        ("min-h", &["min-height"], 'h'),
        ("max-w", &["max-width"], 'w'),
        ("max-h", &["max-height"], 'h'),
        ("size", &["width", "height"], 's'),
    ];
    for (prefix, properties, axis) in families {
        let Some(value) = strip_family(base, prefix) else {
            continue;
        };
        let resolved = size_value(value, axis, theme).ok_or_else(|| unknown(full))?;
        let decls = properties
            .iter()
            .map(|p| (*p, resolved.clone()))
            .collect::<Vec<_>>();
        return Ok(Some(Utility::simple(decls)));
    }
    Ok(None)
}

/// A sizing value: spacing steps, `px`, `full`, `screen` (axis-aware), `auto`,
/// `none`, fractions, the container scale (`sm` -> `var(--container-sm)`),
/// `screen-<bp>` (`var(--breakpoint-md)`), and arbitrary lengths.
fn size_value(value: &str, axis: char, theme: &Theme) -> Option<String> {
    match value {
        "px" => return Some("1px".to_string()),
        "full" => return Some("100%".to_string()),
        "auto" => return Some("auto".to_string()),
        "none" => return Some("none".to_string()),
        "screen" => {
            return Some(match axis {
                'h' => "100vh".to_string(),
                'w' => "100vw".to_string(),
                _ => return None,
            });
        }
        _ => {}
    }
    if let Some(bp) = value.strip_prefix("screen-") {
        let var = format!("--breakpoint-{bp}");
        return theme.contains(&var).then(|| format!("var({var})"));
    }
    let container = format!("--container-{value}");
    if theme.contains(&container) {
        return Some(format!("var({container})"));
    }
    if let Some(percent) = fraction_percent(value) {
        return Some(percent);
    }
    if let Some(inner) = arbitrary_value(value) {
        return Some(inner);
    }
    spacing_value(value, false)
}

/// `grid-template-*` values: a track count or an arbitrary track list.
fn grid_template_value(value: &str) -> Option<String> {
    if !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()) {
        return Some(format!("repeat({value},minmax(0,1fr))"));
    }
    arbitrary_value(value)
}

/// Border utilities: bare side shorthands (`border`, `border-b`), side widths
/// (`border-l-2`), plain widths (`border-2`), side colors (`border-l-amber-500`),
/// and colors (`border-transparent`, `border-[color:var(--x)]`).
fn border_utility(
    base: &str,
    full: &str,
    theme: &Theme,
    tw_props: &mut BTreeSet<TwProp>,
) -> Result<Utility, String> {
    let rest = base.strip_prefix("border").unwrap_or("");
    let rest = rest.strip_prefix('-').unwrap_or(rest);

    // Bare `border` / `border-<side>`.
    if let Some(props) = border_side_decls(rest, "1px") {
        tw_props.insert(TwProp::BorderStyle);
        return Ok(Utility::ranked(props, if rest.is_empty() { 40 } else { 41 }));
    }
    // `border-<side>-<n>` and `border-<n>`.
    if let Some((side, width)) = rest.split_once('-')
        && width.bytes().all(|b| b.is_ascii_digit())
        && !width.is_empty()
        && let Some(props) = border_side_decls(side, &px_width(width))
    {
        tw_props.insert(TwProp::BorderStyle);
        return Ok(Utility::ranked(props, 41));
    }
    if rest.bytes().all(|b| b.is_ascii_digit()) && !rest.is_empty() {
        tw_props.insert(TwProp::BorderStyle);
        let props = border_side_decls("", &px_width(rest)).unwrap();
        return Ok(Utility::ranked(props, 40));
    }
    // `border-<side>-<color>`.
    if let Some((side, color)) = rest.split_once('-')
        && let Some(property) = border_side_color_property(side)
        && let Some(value) = color_value(color, theme)
    {
        return Ok(Utility::ranked(vec![(property, value)], 43));
    }
    // `border-<color>`.
    let value = color_value(rest, theme).ok_or_else(|| unknown(full))?;
    Ok(Utility::ranked(vec![("border-color", value)], 42))
}

fn px_width(n: &str) -> String {
    if n == "0" { "0".to_string() } else { format!("{n}px") }
}

/// The style+width declarations for a border side keyword (`b`, `t`, `l`, `r`,
/// `x`, `y`, or empty for all sides).
fn border_side_decls(side: &str, width: &str) -> Option<Vec<(&'static str, String)>> {
    let (style_prop, width_prop) = match side {
        "" => ("border-style", "border-width"),
        "t" => ("border-top-style", "border-top-width"),
        "r" => ("border-right-style", "border-right-width"),
        "b" => ("border-bottom-style", "border-bottom-width"),
        "l" => ("border-left-style", "border-left-width"),
        "x" => ("border-inline-style", "border-inline-width"),
        "y" => ("border-block-style", "border-block-width"),
        _ => return None,
    };
    Some(vec![
        (style_prop, "var(--tw-border-style)".to_string()),
        (width_prop, width.to_string()),
    ])
}

fn border_side_color_property(side: &str) -> Option<&'static str> {
    Some(match side {
        "t" => "border-top-color",
        "r" => "border-right-color",
        "b" => "border-bottom-color",
        "l" => "border-left-color",
        "x" => "border-inline-color",
        "y" => "border-block-color",
        _ => return None,
    })
}

/// Ring utilities.
fn ring_utility(
    base: &str,
    full: &str,
    theme: &Theme,
    tw_props: &mut BTreeSet<TwProp>,
) -> Result<Utility, String> {
    let rest = base.strip_prefix("ring").unwrap_or("");
    let rest = rest.strip_prefix('-').unwrap_or(rest);
    if rest.is_empty() || (rest.bytes().all(|b| b.is_ascii_digit()) && !rest.is_empty()) {
        let width = if rest.is_empty() { "1" } else { rest };
        register_shadow_group(tw_props);
        return Ok(Utility::simple(vec![
            (
                "--tw-ring-shadow",
                format!(
                    "var(--tw-ring-inset,) 0 0 0 calc({width}px + var(--tw-ring-offset-width)) var(--tw-ring-color,currentcolor)"
                ),
            ),
            ("box-shadow", BOX_SHADOW_CHAIN.to_string()),
        ]));
    }
    if rest == "inset" {
        register_shadow_group(tw_props);
        return Ok(Utility::simple(vec![(
            "--tw-ring-inset",
            "inset".to_string(),
        )]));
    }
    if let Some(n) = rest.strip_prefix("offset-") {
        if n.bytes().all(|b| b.is_ascii_digit()) && !n.is_empty() {
            register_shadow_group(tw_props);
            return Ok(Utility::simple(vec![
                ("--tw-ring-offset-width", format!("{n}px")),
                (
                    "--tw-ring-offset-shadow",
                    "var(--tw-ring-inset,) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color)".to_string(),
                ),
            ]));
        }
        return Err(unknown(full));
    }
    if let Some(value) = color_value(rest, theme) {
        register_shadow_group(tw_props);
        return Ok(Utility::simple(vec![("--tw-ring-color", value)]));
    }
    Err(unknown(full))
}

/// Rewrites each color inside a theme shadow value into
/// `var(--tw-shadow-color, <color>)`, matching the compiled shadow utilities.
fn wrap_shadow_colors(value: &str) -> String {
    let mut out = String::new();
    let mut rest = value;
    while let Some(pos) = rest.find("rgb(") {
        out.push_str(&rest[..pos]);
        let after = &rest[pos..];
        let Some(close) = after.find(')') else {
            out.push_str(after);
            return out;
        };
        let color = &after[..=close];
        out.push_str("var(--tw-shadow-color,");
        out.push_str(color);
        out.push(')');
        rest = &after[close + 1..];
    }
    out.push_str(rest);
    out
}

/// The transition-property composites, verbatim from Tailwind v4.
fn transition_utility(base: &str) -> Option<Vec<(&'static str, String)>> {
    let property = match base {
        "transition" => {
            "color,background-color,border-color,outline-color,text-decoration-color,fill,stroke,--tw-gradient-from,--tw-gradient-via,--tw-gradient-to,opacity,box-shadow,transform,translate,scale,rotate,filter,-webkit-backdrop-filter,backdrop-filter,display,content-visibility,overlay,pointer-events"
        }
        "transition-colors" => {
            "color,background-color,border-color,outline-color,text-decoration-color,fill,stroke,--tw-gradient-from,--tw-gradient-via,--tw-gradient-to"
        }
        "transition-opacity" => "opacity",
        "transition-transform" => "transform,translate,scale,rotate",
        "transition-shadow" => "box-shadow",
        "transition-all" => "all",
        "transition-none" => {
            return Some(vec![("transition-property", "none".to_string())]);
        }
        _ => return None,
    };
    Some(vec![
        ("transition-property", property.to_string()),
        (
            "transition-timing-function",
            "var(--tw-ease,var(--default-transition-timing-function))".to_string(),
        ),
        (
            "transition-duration",
            "var(--tw-duration,var(--default-transition-duration))".to_string(),
        ),
    ])
}

/// Padding/margin/gap utilities over the spacing scale. Margins additionally
/// accept `auto` and negatives. Returns `Ok(None)` when the prefix is not a
/// spacing family, `Err` when it is but the step is invalid.
fn spacing_utility(base: &str, full: &str, negative: bool) -> Result<Option<Utility>, String> {
    let families: [(&str, &str, bool, u16); 15] = [
        ("gap-", "gap", false, 100),
        ("p-", "padding", false, 20),
        ("px-", "padding-inline", false, 21),
        ("py-", "padding-block", false, 22),
        ("pt-", "padding-top", false, 23),
        ("pr-", "padding-right", false, 24),
        ("pb-", "padding-bottom", false, 25),
        ("pl-", "padding-left", false, 26),
        ("m-", "margin", true, 10),
        ("mx-", "margin-inline", true, 11),
        ("my-", "margin-block", true, 12),
        ("mt-", "margin-top", true, 13),
        ("mr-", "margin-right", true, 14),
        ("mb-", "margin-bottom", true, 15),
        ("ml-", "margin-left", true, 16),
    ];
    for (prefix, property, is_margin, rank) in families {
        if let Some(step) = base.strip_prefix(prefix) {
            if negative && !is_margin {
                return Err(unknown(full));
            }
            if is_margin && step == "auto" {
                if negative {
                    return Err(unknown(full));
                }
                return Ok(Some(Utility::ranked(vec![(property, "auto".to_string())], rank)));
            }
            let value = spacing_value(step, negative).ok_or_else(|| unknown(full))?;
            return Ok(Some(Utility::ranked(vec![(property, value)], rank)));
        }
    }
    Ok(None)
}

/// The spacing-scale value for a numeric step (integers or halves like `1.5`),
/// matching Tailwind's compiled output: `0` -> `0`, `1` -> `var(--spacing)`,
/// otherwise `calc(var(--spacing) * n)`.
fn spacing_value(step: &str, negative: bool) -> Option<String> {
    let bytes = step.as_bytes();
    let valid = !step.is_empty()
        && bytes.first().is_some_and(|b| b.is_ascii_digit())
        && bytes.last().is_some_and(|b| b.is_ascii_digit())
        && step.bytes().all(|b| b.is_ascii_digit() || b == b'.')
        && step.bytes().filter(|b| *b == b'.').count() <= 1;
    if !valid {
        return None;
    }
    Some(match (step, negative) {
        ("0", _) => "0".to_string(),
        ("1", false) => "var(--spacing)".to_string(),
        (n, false) => format!("calc(var(--spacing) * {n})"),
        (n, true) => format!("calc(var(--spacing) * -{n})"),
    })
}

/// Resolves a Tailwind color token to a CSS color value: theme colors
/// (`gray-200`, `black`), `transparent`/`current`/`inherit`, arbitrary
/// `[color:…]` values, each with an optional `/<pct>` opacity modifier that
/// compiles to `color-mix(in oklab, …)`.
fn color_value(token: &str, theme: &Theme) -> Option<String> {
    let (token, modifier) = split_color_modifier(token);
    let base = if let Some(inner) = token.strip_prefix("[color:") {
        let inner = inner.strip_suffix(']')?;
        arbitrary_value(&format!("[{inner}]"))?
    } else {
        match token {
            "transparent" => "transparent".to_string(),
            "current" => "currentcolor".to_string(),
            "inherit" => "inherit".to_string(),
            _ => {
                let var = format!("--color-{token}");
                if !theme.contains(&var) {
                    return None;
                }
                format!("var({var})")
            }
        }
    };
    match modifier {
        Some(pct) => Some(format!("color-mix(in oklab, {base} {pct}%, transparent)")),
        None => Some(base),
    }
}

/// Splits a trailing `/<pct>` opacity modifier off a color token (bracket-aware:
/// the `/` must sit outside any `[…]`).
fn split_color_modifier(token: &str) -> (&str, Option<&str>) {
    if let Some(pos) = token.rfind('/') {
        let after_brackets = token[..pos].matches('[').count() == token[..pos].matches(']').count();
        let suffix = &token[pos + 1..];
        let numeric = !suffix.is_empty()
            && suffix.bytes().all(|b| b.is_ascii_digit() || b == b'.');
        if after_brackets && numeric {
            return (&token[..pos], Some(suffix));
        }
    }
    (token, None)
}

fn is_cursor_keyword(kw: &str) -> bool {
    matches!(
        kw,
        "auto" | "default" | "pointer" | "wait" | "text" | "move" | "help" | "not-allowed"
            | "none" | "context-menu" | "progress" | "cell" | "crosshair" | "vertical-text"
            | "alias" | "copy" | "no-drop" | "grab" | "grabbing" | "all-scroll" | "col-resize"
            | "row-resize" | "n-resize" | "e-resize" | "s-resize" | "w-resize" | "ne-resize"
            | "nw-resize" | "se-resize" | "sw-resize" | "ew-resize" | "ns-resize"
            | "nesw-resize" | "nwse-resize" | "zoom-in" | "zoom-out"
    )
}

fn is_font_weight(name: &str) -> bool {
    matches!(
        name,
        "thin"
            | "extralight"
            | "light"
            | "normal"
            | "medium"
            | "semibold"
            | "bold"
            | "extrabold"
            | "black"
    )
}

fn is_text_size(name: &str) -> bool {
    matches!(
        name,
        "xs" | "sm" | "base" | "lg" | "xl" | "2xl" | "3xl" | "4xl" | "5xl" | "6xl" | "7xl"
            | "8xl" | "9xl"
    )
}

/// Escapes a class name for use in a selector: every byte outside
/// `[A-Za-z0-9_-]` is backslash-escaped.
fn escape_class(class: &str) -> String {
    let mut out = String::with_capacity(class.len());
    for c in class.chars() {
        if !(c.is_ascii_alphanumeric() || c == '-' || c == '_') {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

// ---------------------------------------------------------------------------
// User CSS processing
// ---------------------------------------------------------------------------

/// The app's own CSS with `@apply` expanded and the framework import removed.
struct UserCss {
    /// `@custom-variant` definitions: variant name -> `&`-rooted template.
    custom_variants: std::collections::BTreeMap<String, String>,
    /// The `@layer base` body: the app's base rules (with `@apply` expanded) plus
    /// their `dark:` companions as `@media (prefers-color-scheme:dark)` rules
    /// (or the custom `dark` template's selector form when one is defined).
    base_layer: String,
    /// Plain (unlayered) rules passed through after the layers, in source order.
    postlude: String,
    /// Class names the app's own CSS defines (`.markpad-preview`, …): candidates
    /// matching these are satisfied by the app stylesheet, not utilities.
    defined_classes: BTreeSet<String>,
}

/// Processes the app CSS: strips `@import 'tailwindcss'`, walks each `@layer base`
/// block expanding `@apply` directives, and passes plain top-level rules through.
fn process_user_css(
    css: &str,
    theme: &Theme,
    tw_props: &mut BTreeSet<TwProp>,
) -> Result<UserCss, String> {
    let mut base_layer = String::new();
    let mut postlude = String::new();
    let mut defined_classes = BTreeSet::new();
    let items = parse_top_level(css)?;
    // Variants first: a `@custom-variant` applies to the whole sheet no matter
    // where it appears.
    let mut custom_variants = std::collections::BTreeMap::new();
    for item in &items {
        if let TopItem::CustomVariant { name, template } = item {
            custom_variants.insert(name.clone(), template.clone());
        }
    }
    // The `dark:` companion of an expanded rule: under a custom `dark` variant
    // it is a sibling rule with the template applied to each selector; under
    // the default it is a `prefers-color-scheme` media block.
    let emit_dark = |out: &mut String, selector: &str, dark_rule: &str,
                     custom_variants: &std::collections::BTreeMap<String, String>| {
        if let Some(template) = custom_variants.get("dark") {
            let transformed = selector
                .split(',')
                .map(|part| format!("{}{}", part.trim(), &template[1..]))
                .collect::<Vec<_>>()
                .join(",");
            let body = &dark_rule[dark_rule.find('{').map_or(0, |at| at)..];
            out.push_str(&transformed);
            out.push_str(body);
        } else {
            out.push_str("@media (prefers-color-scheme:dark){");
            out.push_str(dark_rule);
            out.push('}');
        }
    };
    for item in items {
        match item {
            TopItem::Import => {}
            TopItem::CustomVariant { .. } => {}
            TopItem::Verbatim(block) => {
                collect_selector_classes(&block, &mut defined_classes);
                postlude.push_str(&block);
            }
            TopItem::Layer { names, body } => {
                if names.split_whitespace().next() != Some("base") {
                    return Err(format!(
                        "unsupported at-rule `@layer {names}` in Tailwind CSS entry (native compiler handles `@layer base`)"
                    ));
                }
                let rules = parse_rules(&body)?;
                for rule in rules {
                    collect_selector_classes(&rule.selector, &mut defined_classes);
                    let (main, dark) = expand_rule(&rule, theme, tw_props)?;
                    if let Some(main) = main {
                        base_layer.push_str(&main);
                    }
                    if let Some(dark) = dark {
                        emit_dark(&mut base_layer, &rule.selector, &dark, &custom_variants);
                    }
                }
            }
            TopItem::Rule { selector, body } => {
                collect_selector_classes(&selector, &mut defined_classes);
                // Tailwind v4 allows `@apply` in any rule, not just `@layer
                // base` — expand it here too (a literal-only body round-trips
                // unchanged through the same expansion).
                let rule = StyleRule {
                    selector: selector.clone(),
                    body,
                };
                let (main, dark) = expand_rule(&rule, theme, tw_props)?;
                if let Some(main) = main {
                    postlude.push_str(&main);
                }
                if let Some(dark) = dark {
                    emit_dark(&mut postlude, &selector, &dark, &custom_variants);
                }
            }
        }
    }
    Ok(UserCss {
        custom_variants,
        base_layer,
        postlude,
        defined_classes,
    })
}

/// Collects `.class` names appearing in a selector.
fn collect_selector_classes(selector: &str, out: &mut BTreeSet<String>) {
    let bytes = selector.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'.' {
            let mut j = i + 1;
            while j < bytes.len()
                && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'-' || bytes[j] == b'_')
            {
                j += 1;
            }
            if j > i + 1 {
                out.insert(selector[i + 1..j].to_string());
            }
            i = j;
        } else {
            i += 1;
        }
    }
}

enum TopItem {
    Import,
    Layer { names: String, body: String },
    Rule { selector: String, body: String },
    /// `@custom-variant <name> (<template>);` — a user-defined variant whose
    /// template (with `&` for the candidate selector) replaces the built-in
    /// meaning of `<name>:` for both utilities and `@apply` expansion.
    CustomVariant { name: String, template: String },
    /// A top-level block passed through verbatim (`@keyframes`, `@media`,
    /// `@supports`, `@font-face`) — app CSS the compiler has no opinion on.
    Verbatim(String),
}

/// Splits the entry into top-level items: `@import` statements, `@layer …{…}`
/// blocks, and plain style rules (passed through). Errors on any other
/// top-level at-rule.
fn parse_top_level(css: &str) -> Result<Vec<TopItem>, String> {
    let mut items = Vec::new();
    let bytes = css.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        i = skip_ws_and_comments(css, i);
        if i >= bytes.len() {
            break;
        }
        if css[i..].starts_with("@import") {
            let end = css[i..].find(';').map(|rel| i + rel + 1).unwrap_or(bytes.len());
            items.push(TopItem::Import);
            i = end;
            continue;
        }
        if css[i..].starts_with("@layer") {
            let brace = css[i..]
                .find('{')
                .ok_or_else(|| "malformed @layer (no `{`)".to_string())?;
            let names = css[i + 6..i + brace].trim().to_string();
            let (body, end) = read_braced(css, i + brace)?;
            items.push(TopItem::Layer { names, body });
            i = end;
            continue;
        }
        if css[i..].starts_with("@custom-variant") {
            let end = css[i..]
                .find(';')
                .map(|rel| i + rel)
                .ok_or_else(|| "malformed @custom-variant (no `;`)".to_string())?;
            let inner = css[i + "@custom-variant".len()..end].trim();
            let open = inner
                .find('(')
                .ok_or_else(|| format!("malformed @custom-variant `{inner}` (no `(`)"))?;
            let name = inner[..open].trim().to_string();
            let template = inner[open + 1..]
                .strip_suffix(')')
                .ok_or_else(|| format!("malformed @custom-variant `{inner}` (no `)`)"))?
                .trim()
                .to_string();
            if name.is_empty() || template.is_empty() {
                return Err(format!("malformed @custom-variant `{inner}`"));
            }
            if !template.starts_with('&') {
                return Err(format!(
                    "@custom-variant `{name}`: only templates that start with `&` are \
                     supported (got `{template}`)"
                ));
            }
            items.push(TopItem::CustomVariant { name, template });
            i = end + 1;
            continue;
        }
        if ["@keyframes", "@media", "@supports", "@font-face"]
            .iter()
            .any(|at| css[i..].starts_with(at))
        {
            let brace = css[i..]
                .find('{')
                .ok_or_else(|| "malformed at-rule block (no `{`)".to_string())?;
            let (_, end) = read_braced(css, i + brace)?;
            items.push(TopItem::Verbatim(css[i..end].to_string()));
            i = end;
            continue;
        }
        if bytes[i] == b'@' {
            return Err(format!(
                "unsupported top-level CSS construct in Tailwind entry near: {:?}",
                &css[i..(i + 40).min(css.len())]
            ));
        }
        // A plain style rule: selector up to `{`, then its declaration block.
        let brace = css[i..]
            .find('{')
            .ok_or_else(|| "malformed rule (no `{`)".to_string())?;
        let selector = css[i..i + brace]
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let (body, end) = read_braced(css, i + brace)?;
        items.push(TopItem::Rule { selector, body });
        i = end;
    }
    Ok(items)
}

/// A single style rule: a selector and its declaration block text.
struct StyleRule {
    selector: String,
    body: String,
}

/// Parses a `@layer base` body into style rules. Errors on nested at-rules
/// (unsupported inside base for this app).
fn parse_rules(css: &str) -> Result<Vec<StyleRule>, String> {
    let mut rules = Vec::new();
    let bytes = css.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        i = skip_ws_and_comments(css, i);
        if i >= bytes.len() {
            break;
        }
        if bytes[i] == b'@' {
            return Err(format!(
                "unsupported nested at-rule inside @layer base near: {:?}",
                &css[i..(i + 40).min(css.len())]
            ));
        }
        let brace = css[i..]
            .find('{')
            .ok_or_else(|| "malformed rule (no `{`)".to_string())?;
        let selector = css[i..i + brace].split_whitespace().collect::<Vec<_>>().join(" ");
        let (body, end) = read_braced(css, i + brace)?;
        rules.push(StyleRule { selector, body });
        i = end;
    }
    Ok(rules)
}

/// Expands a single base rule. Returns `(main_rule, dark_rule)` where `main_rule`
/// is the selector with its literal declarations plus non-variant `@apply`
/// declarations, and `dark_rule` is the same selector with `dark:`-variant
/// `@apply` declarations (to be wrapped in a dark media query).
fn expand_rule(
    rule: &StyleRule,
    theme: &Theme,
    tw_props: &mut BTreeSet<TwProp>,
) -> Result<(Option<String>, Option<String>), String> {
    let mut main_decls: Vec<String> = Vec::new();
    let mut dark_decls: Vec<String> = Vec::new();

    for statement in split_declarations(&rule.body) {
        let statement = statement.trim();
        if statement.is_empty() {
            continue;
        }
        if let Some(classes) = statement.strip_prefix("@apply") {
            for class in classes.split_whitespace() {
                let mut segments: Vec<&str> = class.split(':').collect();
                let apply_base = segments.pop().unwrap_or(class);
                let mut dark = false;
                for variant in &segments {
                    match *variant {
                        "dark" => dark = true,
                        other => {
                            return Err(format!(
                                "unsupported variant `{other}:` in @apply `{class}`"
                            ));
                        }
                    }
                }
                let utility = generate_utility(apply_base, class, theme, tw_props)?;
                for (prop, value) in utility.decls {
                    let decl = format!("{prop}:{value}");
                    if dark {
                        dark_decls.push(decl);
                    } else {
                        main_decls.push(decl);
                    }
                }
            }
        } else {
            // A literal declaration, kept verbatim.
            main_decls.push(statement.to_string());
        }
    }

    let main = if main_decls.is_empty() {
        None
    } else {
        Some(format!("{}{{{}}}", rule.selector, main_decls.join(";")))
    };
    let dark = if dark_decls.is_empty() {
        None
    } else {
        Some(format!("{}{{{}}}", rule.selector, dark_decls.join(";")))
    };
    Ok((main, dark))
}

/// Splits a declaration block body on top-level `;` (there are no nested braces
/// inside a base rule body here).
fn split_declarations(body: &str) -> Vec<String> {
    body.split(';').map(|s| s.to_string()).collect()
}

/// Reads a `{ … }` block starting at the `{` at `open`. Returns the inner body
/// and the index just past the closing `}`.
fn read_braced(css: &str, open: usize) -> Result<(String, usize), String> {
    let bytes = css.as_bytes();
    debug_assert_eq!(bytes[open], b'{');
    let mut depth = 0i32;
    let mut i = open;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Ok((css[open + 1..i].to_string(), i + 1));
                }
            }
            _ => {}
        }
        i += 1;
    }
    Err("unbalanced braces in CSS".to_string())
}

fn skip_ws_and_comments(css: &str, mut i: usize) -> usize {
    let bytes = css.as_bytes();
    loop {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if css[i..].starts_with("/*")
            && let Some(rel) = css[i..].find("*/")
        {
            i += rel + 2;
            continue;
        }
        break;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidates(list: &[&str]) -> BTreeSet<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn detects_tailwind_entry() {
        assert!(is_tailwind_entry("@import 'tailwindcss' source('../');\n"));
        assert!(is_tailwind_entry("@import \"tailwindcss\";"));
        assert!(!is_tailwind_entry("@import './other.css';"));
        assert!(!is_tailwind_entry(".a{color:red}"));
    }

    #[test]
    fn scans_class_candidates_from_every_context() {
        let mut out = BTreeSet::new();
        scan_class_candidates(r#"<div className="p-2 flex gap-2" />"#, &mut out);
        scan_class_candidates(r#"className={`px-2 py-1 font-extrabold`}"#, &mut out);
        scan_class_candidates(r#"activeProps={{ className: 'text-black font-bold' }}"#, &mut out);
        scan_class_candidates(r#"className={`p-${x} flex-col`}"#, &mut out);
        assert!(out.contains("p-2"));
        assert!(out.contains("flex"));
        assert!(out.contains("gap-2"));
        assert!(out.contains("px-2"));
        assert!(out.contains("py-1"));
        assert!(out.contains("font-extrabold"));
        assert!(out.contains("text-black"));
        assert!(out.contains("font-bold"));
        assert!(out.contains("flex-col"));
        // The interpolation must not leak a bogus `p-` token.
        assert!(!out.iter().any(|t| t.contains('$')));
    }

    #[test]
    fn class_list_word_boundary() {
        let mut out = BTreeSet::new();
        // `classList` and `classNames` must not be mistaken for `class`/`className`.
        scan_class_candidates(r#"element.classList = "foo";"#, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn scans_const_bindings_ternaries_and_templates() {
        let source = r#"
            const buttonBase =
              "inline-flex items-center rounded-md px-2.5 disabled:opacity-50";
            const buttonActive = `${buttonBase} bg-transparent`;
            const layoutClass = isSplit
              ? "flex flex-col md:flex-row"
              : "flex flex-col h-full";
            const isSplit = mode === "split";
            function App({ active }) {
              return (
                <div className={layoutClass}>
                  <button className={active ? buttonActive : buttonBase} />
                  <section className={`${pane}${hidden ? " hidden" : ""}`} />
                </div>
              );
            }
            const pane = "flex min-w-0";
        "#;
        let mut out = BTreeSet::new();
        scan_class_candidates(source, &mut out);
        // From const string bindings referenced by className.
        assert!(out.contains("inline-flex"));
        assert!(out.contains("px-2.5"));
        assert!(out.contains("disabled:opacity-50"));
        // Through the template that references another const.
        assert!(out.contains("bg-transparent"));
        // Ternary initializer branches.
        assert!(out.contains("md:flex-row"));
        assert!(out.contains("h-full"));
        // Inline ternary inside a template interpolation.
        assert!(out.contains("hidden"));
        // Const declared after its use still resolves.
        assert!(out.contains("min-w-0"));
        // The `mode === "split"` comparison string must NOT leak: `isSplit` is
        // only a ternary condition and its initializer is not string-shaped.
        assert!(!out.contains("split"));
    }

    #[test]
    fn custom_variant_overrides_dark_for_utilities_apply_and_top_level_rules() {
        let css = "@import 'tailwindcss';\n\
                   @custom-variant dark (&:where(.dark, .dark *));\n\
                   html, body { @apply bg-white dark:bg-gray-900; }\n\
                   @keyframes pop { 0% { transform: scale(0.7); } }\n\
                   .animate-pop { animation: pop 0.2s; }\n";
        let mut candidates = BTreeSet::new();
        candidates.insert("dark:text-gray-100".to_string());
        candidates.insert("animate-pop".to_string());
        let out = compile(css, &candidates).unwrap();
        assert!(
            out.contains(".dark\\:text-gray-100:where(.dark, .dark *)"),
            "utility uses the custom selector: {out}"
        );
        assert!(
            out.contains("html:where(.dark, .dark *)"),
            "@apply dark companion uses the custom selector: {out}"
        );
        assert!(!out.contains("prefers-color-scheme"), "no media dark: {out}");
        assert!(out.contains("@keyframes pop"), "verbatim keyframes survive: {out}");
        assert!(out.contains(".animate-pop"), "plain rule survives: {out}");
    }

    #[test]
    fn compiles_the_reference_utilities() {
        let css = "@import 'tailwindcss' source('../');\n";
        let out = compile(css, &candidates(&["p-2", "flex", "gap-2"])).unwrap();
        assert!(out.contains(".p-2{padding:calc(var(--spacing) * 2)}"));
        assert!(out.contains(".flex{display:flex}"));
        assert!(out.contains(".gap-2{gap:calc(var(--spacing) * 2)}"));
        // The framework import must not survive.
        assert!(!out.contains("@import"));
        assert!(!out.contains("tailwindcss'"));
    }

    #[test]
    fn font_weight_emits_registered_property() {
        let out = compile("@import 'tailwindcss';", &candidates(&["font-black"])).unwrap();
        assert!(out.contains(
            ".font-black{--tw-font-weight:var(--font-weight-black);font-weight:var(--font-weight-black)}"
        ));
        assert!(out.contains("@property --tw-font-weight{syntax:\"*\";inherits:false}"));
        assert!(out.contains("--tw-font-weight:initial"));
        assert!(out.contains("--font-weight-black:900"));
    }

    #[test]
    fn py_1_uses_the_base_spacing_variable() {
        let out = compile("@import 'tailwindcss';", &candidates(&["py-1"])).unwrap();
        assert!(out.contains(".py-1{padding-block:var(--spacing)}"));
        assert!(out.contains("--spacing:0.25rem") || out.contains("--spacing:.25rem"));
    }

    #[test]
    fn text_size_emits_font_size_and_line_height() {
        let out = compile("@import 'tailwindcss';", &candidates(&["text-lg"])).unwrap();
        assert!(out.contains(
            ".text-lg{font-size:var(--text-lg);line-height:var(--tw-leading,var(--text-lg--line-height))}"
        ));
        assert!(out.contains("--text-lg:1.125rem"));
        assert!(out.contains("--text-lg--line-height"));
    }

    #[test]
    fn dark_and_hover_variants_bucket_into_media_queries() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&["hover:text-blue-600", "dark:bg-gray-700"]),
        )
        .unwrap();
        assert!(out.contains(
            "@media (hover:hover){.hover\\:text-blue-600:hover{color:var(--color-blue-600)}}"
        ));
        assert!(out.contains(
            "@media (prefers-color-scheme:dark){.dark\\:bg-gray-700{background-color:var(--color-gray-700)}}"
        ));
    }

    #[test]
    fn space_y_emits_reverse_variable_and_child_selector() {
        let out = compile("@import 'tailwindcss';", &candidates(&["space-y-2"])).unwrap();
        assert!(out.contains(":where(.space-y-2>:not(:last-child)){--tw-space-y-reverse:0;margin-block-start:calc(calc(var(--spacing) * 2) * var(--tw-space-y-reverse));margin-block-end:calc(calc(var(--spacing) * 2) * calc(1 - var(--tw-space-y-reverse)))}"));
        assert!(out.contains("@property --tw-space-y-reverse{syntax:\"*\";inherits:false;initial-value:0}"));
        // Fractional steps are part of the same scale.
        let out = compile("@import 'tailwindcss';", &candidates(&["space-y-1.5"])).unwrap();
        assert!(out.contains(":where(.space-y-1\\.5>:not(:last-child))"));
        assert!(out.contains("calc(var(--spacing) * 1.5)"));
    }

    #[test]
    fn border_side_emits_style_and_width() {
        let out = compile("@import 'tailwindcss';", &candidates(&["border-b"])).unwrap();
        assert!(out.contains(
            ".border-b{border-bottom-style:var(--tw-border-style);border-bottom-width:1px}"
        ));
        assert!(out.contains("@property --tw-border-style{syntax:\"*\";inherits:false;initial-value:solid}"));
    }

    #[test]
    fn border_widths_colors_and_arbitrary_values() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&[
                "border-l-2",
                "md:border-t-0",
                "border-l-amber-500",
                "border-transparent",
                "border-[color:var(--border)]",
            ]),
        )
        .unwrap();
        assert!(out.contains(
            ".border-l-2{border-left-style:var(--tw-border-style);border-left-width:2px}"
        ));
        assert!(out.contains(
            "@media (width>=48rem){.md\\:border-t-0{border-top-style:var(--tw-border-style);border-top-width:0}}"
        ));
        assert!(out.contains(".border-l-amber-500{border-left-color:var(--color-amber-500)}"));
        assert!(out.contains(".border-transparent{border-color:transparent}"));
        assert!(out.contains(".border-\\[color\\:var\\(--border\\)\\]{border-color:var(--border)}"));
    }

    #[test]
    fn position_keywords_offsets_fractions_and_negatives() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&[
                "absolute",
                "relative",
                "fixed",
                "inset-0",
                "top-1/2",
                "left-0",
                "-left-1",
                "-left-[100vw]",
                "inset-y-0",
                "z-50",
            ]),
        )
        .unwrap();
        assert!(out.contains(".absolute{position:absolute}"));
        assert!(out.contains(".relative{position:relative}"));
        assert!(out.contains(".fixed{position:fixed}"));
        assert!(out.contains(".inset-0{inset:0}"));
        assert!(out.contains(".top-1\\/2{top:50%}"));
        assert!(out.contains(".left-0{left:0}"));
        assert!(out.contains(".-left-1{left:calc(var(--spacing) * -1)}"));
        assert!(out.contains(".-left-\\[100vw\\]{left:calc(100vw * -1)}"));
        assert!(out.contains(".inset-y-0{inset-block:0}"));
        assert!(out.contains(".z-50{z-index:50}"));
    }

    #[test]
    fn sizing_family_lengths_keywords_and_arbitrary() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&[
                "w-full", "w-px", "w-screen", "w-0.5", "h-screen", "h-1.5", "h-px",
                "min-w-0", "min-w-[10rem]", "min-h-0", "max-w-sm", "max-w-[18rem]",
                "max-w-screen-md", "size-5",
            ]),
        )
        .unwrap();
        assert!(out.contains(".w-full{width:100%}"));
        assert!(out.contains(".w-px{width:1px}"));
        assert!(out.contains(".w-screen{width:100vw}"));
        assert!(out.contains(".w-0\\.5{width:calc(var(--spacing) * 0.5)}"));
        assert!(out.contains(".h-screen{height:100vh}"));
        assert!(out.contains(".h-1\\.5{height:calc(var(--spacing) * 1.5)}"));
        assert!(out.contains(".min-w-0{min-width:0}"));
        assert!(out.contains(".min-w-\\[10rem\\]{min-width:10rem}"));
        assert!(out.contains(".max-w-sm{max-width:var(--container-sm)}"));
        assert!(out.contains("--container-sm:24rem"));
        assert!(out.contains(".max-w-\\[18rem\\]{max-width:18rem}"));
        assert!(out.contains(".max-w-screen-md{max-width:var(--breakpoint-md)}"));
        assert!(out.contains(
            ".size-5{width:calc(var(--spacing) * 5);height:calc(var(--spacing) * 5)}"
        ));
    }

    #[test]
    fn translate_fractions_register_properties() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&["-translate-y-1/2", "translate-x-2", "-translate-x-1/2"]),
        )
        .unwrap();
        assert!(out.contains(
            ".-translate-y-1\\/2{--tw-translate-y:calc(calc(1 / 2 * 100%) * -1);translate:var(--tw-translate-x) var(--tw-translate-y)}"
        ));
        assert!(out.contains(
            ".-translate-x-1\\/2{--tw-translate-x:calc(calc(1 / 2 * 100%) * -1);translate:var(--tw-translate-x) var(--tw-translate-y)}"
        ));
        assert!(out.contains(
            ".translate-x-2{--tw-translate-x:calc(var(--spacing) * 2);translate:var(--tw-translate-x) var(--tw-translate-y)}"
        ));
        assert!(out.contains("@property --tw-translate-x{syntax:\"*\";inherits:false;initial-value:0}"));
        assert!(out.contains("--tw-translate-z:0"));
    }

    #[test]
    fn margin_auto_negatives_and_all_sides() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&["m-auto", "ml-auto", "mx-2", "my-1.5", "mr-1", "mb-4", "-mt-2"]),
        )
        .unwrap();
        assert!(out.contains(".m-auto{margin:auto}"));
        assert!(out.contains(".ml-auto{margin-left:auto}"));
        assert!(out.contains(".mx-2{margin-inline:calc(var(--spacing) * 2)}"));
        assert!(out.contains(".my-1\\.5{margin-block:calc(var(--spacing) * 1.5)}"));
        assert!(out.contains(".mr-1{margin-right:var(--spacing)}"));
        assert!(out.contains(".mb-4{margin-bottom:calc(var(--spacing) * 4)}"));
        assert!(out.contains(".-mt-2{margin-top:calc(var(--spacing) * -2)}"));
    }

    #[test]
    fn gap_axes_and_grid_templates() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&["gap-x-3", "gap-y-2", "gap-0.5", "grid-cols-2", "grid-cols-[auto_1fr]"]),
        )
        .unwrap();
        assert!(out.contains(".gap-x-3{column-gap:calc(var(--spacing) * 3)}"));
        assert!(out.contains(".gap-y-2{row-gap:calc(var(--spacing) * 2)}"));
        assert!(out.contains(".gap-0\\.5{gap:calc(var(--spacing) * 0.5)}"));
        assert!(out.contains(".grid-cols-2{grid-template-columns:repeat(2,minmax(0,1fr))}"));
        assert!(out.contains(".grid-cols-\\[auto_1fr\\]{grid-template-columns:auto 1fr}"));
    }

    #[test]
    fn color_values_arbitrary_and_opacity_modifiers() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&[
                "bg-[color:var(--panel)]",
                "bg-amber-500/10",
                "bg-transparent",
                "text-[color:var(--muted)]",
                "text-[11px]",
                "accent-[color:var(--accent)]",
            ]),
        )
        .unwrap();
        assert!(out.contains(
            ".bg-\\[color\\:var\\(--panel\\)\\]{background-color:var(--panel)}"
        ));
        assert!(out.contains(
            ".bg-amber-500\\/10{background-color:color-mix(in oklab, var(--color-amber-500) 10%, transparent)}"
        ));
        assert!(out.contains(".bg-transparent{background-color:transparent}"));
        assert!(out.contains(".text-\\[color\\:var\\(--muted\\)\\]{color:var(--muted)}"));
        assert!(out.contains(".text-\\[11px\\]{font-size:11px}"));
        assert!(out.contains(
            ".accent-\\[color\\:var\\(--accent\\)\\]{accent-color:var(--accent)}"
        ));
        assert!(out.contains("--color-amber-500:"));
    }

    #[test]
    fn typography_families() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&[
                "text-left",
                "text-center",
                "leading-tight",
                "leading-none",
                "tracking-wide",
                "tabular-nums",
                "truncate",
                "line-clamp-2",
                "break-all",
                "italic",
            ]),
        )
        .unwrap();
        assert!(out.contains(".text-left{text-align:left}"));
        assert!(out.contains(".text-center{text-align:center}"));
        assert!(out.contains(
            ".leading-tight{--tw-leading:var(--leading-tight);line-height:var(--leading-tight)}"
        ));
        assert!(out.contains(".leading-none{--tw-leading:1;line-height:1}"));
        assert!(out.contains(
            ".tracking-wide{--tw-tracking:var(--tracking-wide);letter-spacing:var(--tracking-wide)}"
        ));
        assert!(out.contains(
            ".tabular-nums{--tw-numeric-spacing:tabular-nums;font-variant-numeric:var(--tw-ordinal,) var(--tw-slashed-zero,) var(--tw-numeric-figure,) var(--tw-numeric-spacing,) var(--tw-numeric-fraction,)}"
        ));
        assert!(out.contains(
            ".truncate{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}"
        ));
        assert!(out.contains(
            ".line-clamp-2{overflow:hidden;display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:2}"
        ));
        assert!(out.contains(".break-all{word-break:break-all}"));
        assert!(out.contains(".italic{font-style:italic}"));
        assert!(out.contains("@property --tw-numeric-spacing{syntax:\"*\";inherits:false}"));
    }

    #[test]
    fn shadows_rings_and_outline_share_the_box_shadow_group() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&["shadow-lg", "ring-2", "ring-inset", "outline", "outline-none"]),
        )
        .unwrap();
        assert!(out.contains(
            ".shadow-lg{--tw-shadow:0 10px 15px -3px var(--tw-shadow-color,rgb(0 0 0 / 0.1)), 0 4px 6px -4px var(--tw-shadow-color,rgb(0 0 0 / 0.1));box-shadow:var(--tw-inset-shadow), var(--tw-inset-ring-shadow), var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow)}"
        ));
        assert!(out.contains(
            ".ring-2{--tw-ring-shadow:var(--tw-ring-inset,) 0 0 0 calc(2px + var(--tw-ring-offset-width)) var(--tw-ring-color,currentcolor);box-shadow:"
        ));
        assert!(out.contains(".ring-inset{--tw-ring-inset:inset}"));
        assert!(out.contains(
            ".outline{outline-style:var(--tw-outline-style);outline-width:1px}"
        ));
        assert!(out.contains(".outline-none{--tw-outline-style:none;outline-style:none}"));
        assert!(out.contains("@property --tw-ring-offset-color{syntax:\"*\";inherits:false;initial-value:#fff}"));
        assert!(out.contains("--tw-ring-shadow:0 0 #0000"));
    }

    #[test]
    fn transition_cursor_select_and_pointer_events() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&[
                "transition-colors",
                "transition-opacity",
                "cursor-pointer",
                "cursor-col-resize",
                "select-none",
                "pointer-events-none",
                "opacity-45",
                "opacity-0",
            ]),
        )
        .unwrap();
        assert!(out.contains(
            ".transition-colors{transition-property:color,background-color,border-color,outline-color,text-decoration-color,fill,stroke,--tw-gradient-from,--tw-gradient-via,--tw-gradient-to;transition-timing-function:var(--tw-ease,var(--default-transition-timing-function));transition-duration:var(--tw-duration,var(--default-transition-duration))}"
        ));
        assert!(out.contains(".transition-opacity{transition-property:opacity;"));
        assert!(out.contains(".cursor-pointer{cursor:pointer}"));
        assert!(out.contains(".cursor-col-resize{cursor:col-resize}"));
        assert!(out.contains(".select-none{-webkit-user-select:none;user-select:none}"));
        assert!(out.contains(".pointer-events-none{pointer-events:none}"));
        assert!(out.contains(".opacity-45{opacity:.45}"));
        assert!(out.contains(".opacity-0{opacity:0}"));
        assert!(out.contains("--default-transition-duration:"));
    }

    #[test]
    fn rounded_scale_full_and_sr_only() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&["rounded", "rounded-full", "rounded-md", "sr-only"]),
        )
        .unwrap();
        assert!(out.contains(".rounded{border-radius:0.25rem}"));
        assert!(out.contains(".rounded-full{border-radius:3.40282e38px}"));
        assert!(out.contains(".rounded-md{border-radius:var(--radius-md)}"));
        assert!(out.contains(
            ".sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip-path:inset(50%);white-space:nowrap;border-width:0}"
        ));
    }

    #[test]
    fn pseudo_variants_focus_disabled_and_combinations() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&[
                "focus:outline-none",
                "focus-visible:ring-2",
                "disabled:opacity-50",
                "disabled:hover:bg-transparent",
            ]),
        )
        .unwrap();
        assert!(out.contains(
            ".focus\\:outline-none:focus{--tw-outline-style:none;outline-style:none}"
        ));
        assert!(out.contains(".focus-visible\\:ring-2:focus-visible{--tw-ring-shadow:"));
        assert!(out.contains(".disabled\\:opacity-50:disabled{opacity:.5}"));
        assert!(out.contains(
            "@media (hover:hover){.disabled\\:hover\\:bg-transparent:disabled:hover{background-color:transparent}}"
        ));
        // The plain-disabled rule must precede the hover-media companion.
        let disabled = out.find(".disabled\\:opacity-50:disabled").unwrap();
        let disabled_hover = out.find(".disabled\\:hover\\:bg-transparent").unwrap();
        assert!(disabled < disabled_hover);
    }

    #[test]
    fn before_backdrop_and_group_hover_variants() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&[
                "before:absolute",
                "before:-left-1",
                "before:content-['']",
                "backdrop:bg-black/40",
                "group-hover:opacity-100",
                "group",
            ]),
        )
        .unwrap();
        assert!(out.contains(
            ".before\\:absolute:before{content:var(--tw-content);position:absolute}"
        ));
        assert!(out.contains(
            ".before\\:-left-1:before{content:var(--tw-content);left:calc(var(--spacing) * -1)}"
        ));
        assert!(out.contains(
            ".before\\:content-\\[\\'\\'\\]:before{--tw-content:'';content:var(--tw-content)}"
        ));
        assert!(out.contains(
            ".backdrop\\:bg-black\\/40::backdrop{background-color:color-mix(in oklab, var(--color-black) 40%, transparent)}"
        ));
        assert!(out.contains(
            "@media (hover:hover){.group-hover\\:opacity-100:is(:where(.group):hover *){opacity:1}}"
        ));
        // `group` itself is a marker class: no rule, no error.
        assert!(!out.contains(".group{"));
        assert!(out.contains("@property --tw-content{syntax:\"*\";inherits:false;initial-value:\"\"}"));
    }

    #[test]
    fn breakpoint_variants_use_theme_widths() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&["md:flex-row", "sm:px-2", "lg:grid-cols-3"]),
        )
        .unwrap();
        assert!(out.contains("@media (width>=48rem){.md\\:flex-row{flex-direction:row}}"));
        assert!(out.contains(
            "@media (width>=40rem){.sm\\:px-2{padding-inline:calc(var(--spacing) * 2)}}"
        ));
        assert!(out.contains(
            "@media (width>=64rem){.lg\\:grid-cols-3{grid-template-columns:repeat(3,minmax(0,1fr))}}"
        ));
        // Breakpoint blocks come in ascending width order.
        let sm = out.find("width>=40rem").unwrap();
        let md = out.find("width>=48rem").unwrap();
        let lg = out.find("width>=64rem").unwrap();
        assert!(sm < md && md < lg);
    }

    #[test]
    fn plain_user_rules_pass_through_after_utilities() {
        let css = "@import 'tailwindcss';\n\
            :root { --bg: #ffffff; color-scheme: light; }\n\
            .no-scrollbar { scrollbar-width: none; }\n\
            .no-scrollbar::-webkit-scrollbar { display: none; }\n\
            .markpad-preview h1 { font-size: 1.75rem; }\n";
        let out = compile(css, &candidates(&["flex", "no-scrollbar", "markpad-preview"])).unwrap();
        assert!(out.contains(":root{--bg: #ffffff;color-scheme: light}"));
        assert!(out.contains(".no-scrollbar{scrollbar-width: none}"));
        assert!(out.contains(".no-scrollbar::-webkit-scrollbar{display: none}"));
        assert!(out.contains(".markpad-preview h1{font-size: 1.75rem}"));
        // The user rules sit after the utilities layer (unlayered wins).
        let utilities = out.find("@layer utilities{").unwrap();
        let user = out.find(".no-scrollbar{").unwrap();
        assert!(user > utilities);
        // App-defined classes are satisfied by the app CSS: no utility, no error.
        assert!(!out.contains(".markpad-preview{display"));
    }

    #[test]
    fn expands_apply_in_base_layer_with_dark_split() {
        let css = "@import 'tailwindcss' source('../');\n\
            @layer base {\n\
              html, body { @apply text-gray-900 bg-gray-50 dark:bg-gray-950 dark:text-gray-200; }\n\
            }\n";
        let out = compile(css, &candidates(&[])).unwrap();
        // Non-variant applies land inline on the selector.
        assert!(out.contains("html, body{color:var(--color-gray-900);background-color:var(--color-gray-50)}") ||
                out.contains("html, body{background-color:var(--color-gray-50);color:var(--color-gray-900)}"));
        // The dark applies land in a dark media rule with the same selector.
        assert!(out.contains("@media (prefers-color-scheme:dark){html, body{"));
        assert!(out.contains("--color-gray-950"));
        assert!(out.contains("--color-gray-200"));
        // The referenced gray tokens are emitted in the theme layer.
        assert!(out.contains("--color-gray-50:"));
        assert!(out.contains("--color-gray-900:"));
    }

    #[test]
    fn literal_base_declarations_pass_through() {
        let css = "@import 'tailwindcss';\n\
            @layer base { .using-mouse * { outline: none !important; } }\n";
        let out = compile(css, &candidates(&[])).unwrap();
        assert!(out.contains(".using-mouse *{outline: none !important}"));
    }

    #[test]
    fn unknown_utility_is_a_hard_error_naming_the_token() {
        let err = compile("@import 'tailwindcss';", &candidates(&["p-bogus"])).unwrap_err();
        assert!(err.contains("p-bogus"), "error must name the token: {err}");

        let err = compile("@import 'tailwindcss';", &candidates(&["bg-plaid-500"])).unwrap_err();
        assert!(err.contains("bg-plaid-500"), "error must name the token: {err}");

        let err = compile("@import 'tailwindcss';", &candidates(&["text-gray-1000"])).unwrap_err();
        assert!(err.contains("text-gray-1000"), "error must name the token: {err}");

        // All failures are reported together, not one at a time.
        let err = compile(
            "@import 'tailwindcss';",
            &candidates(&["p-bogus", "bg-plaid-500"]),
        )
        .unwrap_err();
        assert!(err.contains("p-bogus") && err.contains("bg-plaid-500"));
    }

    #[test]
    fn unknown_variant_is_a_hard_error() {
        let err =
            compile("@import 'tailwindcss';", &candidates(&["aria-checked:flex"])).unwrap_err();
        assert!(err.contains("aria-checked"), "error must name the variant: {err}");
    }

    #[test]
    fn preflight_and_banner_are_present_no_import_survives() {
        let out = compile("@import 'tailwindcss' source('../');", &candidates(&["flex"])).unwrap();
        assert!(out.starts_with("/*! tailwindcss v4.3.3"));
        assert!(out.contains("@layer base{"));
        assert!(out.contains("box-sizing:border-box"));
        assert!(!out.contains("tailwindcss'"));
        assert!(!out.to_lowercase().contains("@import"));
    }
}
