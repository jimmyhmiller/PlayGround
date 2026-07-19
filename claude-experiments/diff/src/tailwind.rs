//! Native Tailwind v4 CSS compilation.
//!
//! TanStack Start's `src/styles/app.css` is a Tailwind v4 entry point
//! (`@import 'tailwindcss'`, `@layer base`, `@apply`, and `dark:`/`hover:`
//! variants). The reference Vite build runs `@tailwindcss/vite`, which compiles
//! it into a single extracted stylesheet: the v4 preflight, the theme tokens the
//! app actually references, the app's own base rules with `@apply` expanded, and
//! one utility rule per class the app uses. Diffpack used to copy the raw source
//! through the `?url` loader, so the browser fetched `@import 'tailwindcss'` and
//! 404'd on `/assets/tailwindcss` and the app rendered unstyled.
//!
//! This module is a native Rust implementation of that compile. It is a *general*
//! utility engine driven by faithful Tailwind v4 reference data — the published
//! default theme ([`THEME_CSS`], verbatim from the `tailwindcss` package) and the
//! resolved preflight ([`PREFLIGHT_CSS`]) — never a lookup table of the app's
//! specific classes. A class the app uses that the engine does not yet handle is
//! a hard, specific error naming the token; it is never silently dropped.
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
/// (see [`scan_class_candidates`]). Every candidate must resolve to a utility or
/// this returns a hard error naming the token.
pub fn compile(css: &str, candidate_classes: &BTreeSet<String>) -> Result<String, String> {
    let theme = Theme::parse(THEME_CSS);

    // 1. Generate one rule per candidate utility, bucketed by variant.
    let mut base_utilities: Vec<String> = Vec::new();
    let mut hover_utilities: Vec<String> = Vec::new();
    let mut dark_utilities: Vec<String> = Vec::new();
    let mut tw_props: BTreeSet<TwProp> = BTreeSet::new();

    for class in candidate_classes {
        let rule = render_utility(class, &mut tw_props)?;
        match rule.bucket {
            VariantBucket::Base => base_utilities.push(rule.css),
            VariantBucket::Hover => hover_utilities.push(rule.css),
            VariantBucket::Dark => dark_utilities.push(rule.css),
        }
    }

    // 2. Process the app's own CSS: strip the framework import, expand `@apply`
    //    inside `@layer base`, split `dark:` applies into a dark media rule.
    let user = process_user_css(css, &mut tw_props)?;

    // 3. Determine which theme tokens the generated CSS references.
    let mut referenced: BTreeSet<String> = BTreeSet::new();
    collect_theme_vars(&base_utilities, &theme, &mut referenced);
    collect_theme_vars(&hover_utilities, &theme, &mut referenced);
    collect_theme_vars(&dark_utilities, &theme, &mut referenced);
    collect_theme_vars(std::slice::from_ref(&user.base_layer), &theme, &mut referenced);
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
            &ordered_tw_props(&tw_props)
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
    for rule in &base_utilities {
        out.push_str(rule);
    }
    if !hover_utilities.is_empty() {
        out.push_str("@media (hover:hover){");
        for rule in &hover_utilities {
            out.push_str(rule);
        }
        out.push('}');
    }
    if !dark_utilities.is_empty() {
        out.push_str("@media (prefers-color-scheme:dark){");
        for rule in &dark_utilities {
            out.push_str(rule);
        }
        out.push('}');
    }
    out.push('}');

    for prop in ordered_tw_props(&tw_props) {
        out.push_str(prop.property_declaration());
    }

    Ok(out)
}

/// Scans a JavaScript/TypeScript/JSX source for utility class candidates. Extracts
/// the string content of every `className`/`class` attribute or object property
/// (double-quoted, single-quoted, or template literal), splitting on whitespace.
/// Template interpolations (`${...}`) are treated as token boundaries. This is the
/// precise, class-scoped candidate set: every extracted token is a real class the
/// app applies, so an unhandled one is a genuine gap, not a false positive.
pub fn scan_class_candidates(source: &str, out: &mut BTreeSet<String>) {
    let bytes = source.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Find the next `class`/`className` identifier used as an attribute or
        // object property.
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
        // Optional JSX expression brace.
        if k < bytes.len() && bytes[k] == b'{' {
            k = skip_ws(bytes, k + 1);
        }
        if k >= bytes.len() {
            break;
        }
        let quote = bytes[k];
        if quote != b'"' && quote != b'\'' && quote != b'`' {
            i = start + 5;
            continue;
        }
        // Read the string literal body until the matching unescaped quote.
        let mut body = String::new();
        let mut p = k + 1;
        let mut depth = 0i32; // `${...}` interpolation depth in template literals.
        while p < bytes.len() {
            let c = bytes[p];
            if depth > 0 {
                if c == b'{' {
                    depth += 1;
                } else if c == b'}' {
                    depth -= 1;
                }
                p += 1;
                continue;
            }
            if c == b'\\' {
                // Skip the escaped character.
                p += 2;
                continue;
            }
            if quote == b'`' && c == b'$' && p + 1 < bytes.len() && bytes[p + 1] == b'{' {
                body.push(' ');
                depth = 1;
                p += 2;
                continue;
            }
            if c == quote {
                break;
            }
            body.push(c as char);
            p += 1;
        }
        for token in body.split_whitespace() {
            if !token.is_empty() && !token.contains('$') {
                out.insert(token.to_string());
            }
        }
        i = p + 1;
    }
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
fn collect_theme_vars(chunks: &[String], theme: &Theme, out: &mut BTreeSet<String>) {
    for chunk in chunks {
        let mut rest = chunk.as_str();
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
}

/// A registered `--tw-*` custom property a utility depends on.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum TwProp {
    SpaceYReverse,
    BorderStyle,
    FontWeight,
}

impl TwProp {
    fn layer_declaration(self) -> &'static str {
        match self {
            TwProp::SpaceYReverse => "--tw-space-y-reverse:0",
            TwProp::BorderStyle => "--tw-border-style:solid",
            TwProp::FontWeight => "--tw-font-weight:initial",
        }
    }

    fn property_declaration(self) -> &'static str {
        match self {
            TwProp::SpaceYReverse => {
                "@property --tw-space-y-reverse{syntax:\"*\";inherits:false;initial-value:0}"
            }
            TwProp::BorderStyle => {
                "@property --tw-border-style{syntax:\"*\";inherits:false;initial-value:solid}"
            }
            TwProp::FontWeight => "@property --tw-font-weight{syntax:\"*\";inherits:false}",
        }
    }
}

/// Canonical Tailwind order for the registered properties.
fn ordered_tw_props(set: &BTreeSet<TwProp>) -> Vec<TwProp> {
    [TwProp::SpaceYReverse, TwProp::BorderStyle, TwProp::FontWeight]
        .into_iter()
        .filter(|prop| set.contains(prop))
        .collect()
}

#[derive(Clone, Copy)]
enum VariantBucket {
    Base,
    Hover,
    Dark,
}

struct RenderedRule {
    css: String,
    bucket: VariantBucket,
}

/// Renders a full utility class (with any `dark:`/`hover:` variant) into a CSS
/// rule string and the bucket it belongs to. Errors if the token is not a
/// recognized utility.
fn render_utility(class: &str, tw_props: &mut BTreeSet<TwProp>) -> Result<RenderedRule, String> {
    let mut segments: Vec<&str> = class.split(':').collect();
    let base = segments
        .pop()
        .ok_or_else(|| format!("empty class candidate `{class}`"))?;
    let mut bucket = VariantBucket::Base;
    let mut hover = false;
    for variant in &segments {
        match *variant {
            "dark" => bucket = VariantBucket::Dark,
            "hover" => {
                hover = true;
                bucket = VariantBucket::Hover;
            }
            other => {
                return Err(format!(
                    "unsupported Tailwind variant `{other}:` in class `{class}` (native compiler supports `dark:` and `hover:`)"
                ));
            }
        }
    }

    let utility = generate_utility(base, class, tw_props)?;
    let escaped = escape_class(class);
    let selector = match utility.selector {
        SelectorKind::Class => {
            if hover {
                format!(".{escaped}:hover")
            } else {
                format!(".{escaped}")
            }
        }
        SelectorKind::SpaceChildren => {
            format!(":where(.{escaped}>:not(:last-child))")
        }
    };
    let body = utility
        .decls
        .iter()
        .map(|(prop, value)| format!("{prop}:{value}"))
        .collect::<Vec<_>>()
        .join(";");
    Ok(RenderedRule {
        css: format!("{selector}{{{body}}}"),
        bucket,
    })
}

enum SelectorKind {
    Class,
    SpaceChildren,
}

struct Utility {
    selector: SelectorKind,
    decls: Vec<(String, String)>,
}

impl Utility {
    fn simple(decls: Vec<(&str, String)>) -> Utility {
        Utility {
            selector: SelectorKind::Class,
            decls: decls
                .into_iter()
                .map(|(prop, value)| (prop.to_string(), value))
                .collect(),
        }
    }
}

/// The general utility generator. `base` is the class with variants stripped;
/// `full` is the original token (for error messages). Returns a hard error naming
/// the token if it matches a known utility family but references an unknown value,
/// or if the family itself is unimplemented.
fn generate_utility(
    base: &str,
    full: &str,
    tw_props: &mut BTreeSet<TwProp>,
) -> Result<Utility, String> {
    // Keyword utilities (no value segment).
    let keyword = match base {
        "block" => Some(vec![("display", "block".to_string())]),
        "inline-block" => Some(vec![("display", "inline-block".to_string())]),
        "inline" => Some(vec![("display", "inline".to_string())]),
        "flex" => Some(vec![("display", "flex".to_string())]),
        "grid" => Some(vec![("display", "grid".to_string())]),
        "hidden" => Some(vec![("display", "none".to_string())]),
        "flex-col" => Some(vec![("flex-direction", "column".to_string())]),
        "flex-row" => Some(vec![("flex-direction", "row".to_string())]),
        "flex-wrap" => Some(vec![("flex-wrap", "wrap".to_string())]),
        "flex-1" => Some(vec![("flex", "1".to_string())]),
        "items-center" => Some(vec![("align-items", "center".to_string())]),
        "items-start" => Some(vec![("align-items", "flex-start".to_string())]),
        "items-end" => Some(vec![("align-items", "flex-end".to_string())]),
        "justify-center" => Some(vec![("justify-content", "center".to_string())]),
        "justify-between" => Some(vec![("justify-content", "space-between".to_string())]),
        "list-disc" => Some(vec![("list-style-type", "disc".to_string())]),
        "list-decimal" => Some(vec![("list-style-type", "decimal".to_string())]),
        "list-none" => Some(vec![("list-style-type", "none".to_string())]),
        "uppercase" => Some(vec![("text-transform", "uppercase".to_string())]),
        "lowercase" => Some(vec![("text-transform", "lowercase".to_string())]),
        "capitalize" => Some(vec![("text-transform", "capitalize".to_string())]),
        "underline" => Some(vec![("text-decoration-line", "underline".to_string())]),
        "line-through" => Some(vec![("text-decoration-line", "line-through".to_string())]),
        "whitespace-nowrap" => Some(vec![("white-space", "nowrap".to_string())]),
        "min-w-0" => Some(vec![("min-width", "0".to_string())]),
        "min-h-0" => Some(vec![("min-height", "0".to_string())]),
        _ => None,
    };
    if let Some(decls) = keyword {
        return Ok(Utility::simple(decls));
    }

    // `border-b` and friends: a border side gets style + width (via --tw-border-style).
    if let Some(side) = base.strip_prefix("border-") {
        if let Some((props, _)) = border_side(side) {
            tw_props.insert(TwProp::BorderStyle);
            return Ok(Utility::simple(props));
        }
        // Otherwise `border-<color>` is a border color utility.
        let value = resolve_color(side).ok_or_else(|| unknown(full))?;
        return Ok(Utility::simple(vec![("border-color", value)]));
    }

    // Spacing: p/px/py/pt/pr/pb/pl/m/gap.
    if let Some(utility) = spacing_utility(base)? {
        return Ok(utility);
    }

    // space-y-<n>: adjacent-sibling margin utility.
    if let Some(n) = base.strip_prefix("space-y-") {
        let value = spacing_value(n).ok_or_else(|| unknown(full))?;
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
        });
    }

    // rounded-<size>: border-radius from the theme radius scale.
    if base == "rounded" {
        return Ok(Utility::simple(vec![("border-radius", "var(--radius-sm)".to_string())]));
    }
    if let Some(size) = base.strip_prefix("rounded-") {
        let var = format!("--radius-{size}");
        if THEME_CSS.contains(&format!("{var}:")) || THEME_CSS.contains(&format!("{var} :")) {
            return Ok(Utility::simple(vec![(
                "border-radius",
                format!("var({var})"),
            )]));
        }
        return Err(unknown(full));
    }

    // bg-<color>.
    if let Some(color) = base.strip_prefix("bg-") {
        let value = resolve_color(color).ok_or_else(|| unknown(full))?;
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

    // text-<size> or text-<color>.
    if let Some(rest) = base.strip_prefix("text-") {
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
        let value = resolve_color(rest).ok_or_else(|| unknown(full))?;
        return Ok(Utility::simple(vec![("color", value)]));
    }

    Err(unknown(full))
}

fn unknown(full: &str) -> String {
    format!(
        "unsupported Tailwind utility class `{full}`: the native compiler does not yet generate it. Extend src/tailwind.rs (do not silently drop it)."
    )
}

/// The declarations for a border side keyword (`b`, `t`, `l`, `r`, `x`, `y`, or
/// empty for all sides). Returns `None` if `side` is not a side keyword (so the
/// caller can try a color).
fn border_side(side: &str) -> Option<(Vec<(&'static str, String)>, ())> {
    let props: Vec<(&'static str, String)> = match side {
        "" => vec![
            ("border-style", "var(--tw-border-style)".to_string()),
            ("border-width", "1px".to_string()),
        ],
        "b" => vec![
            ("border-bottom-style", "var(--tw-border-style)".to_string()),
            ("border-bottom-width", "1px".to_string()),
        ],
        "t" => vec![
            ("border-top-style", "var(--tw-border-style)".to_string()),
            ("border-top-width", "1px".to_string()),
        ],
        "l" => vec![
            ("border-left-style", "var(--tw-border-style)".to_string()),
            ("border-left-width", "1px".to_string()),
        ],
        "r" => vec![
            ("border-right-style", "var(--tw-border-style)".to_string()),
            ("border-right-width", "1px".to_string()),
        ],
        "x" => vec![
            ("border-inline-style", "var(--tw-border-style)".to_string()),
            ("border-inline-width", "1px".to_string()),
        ],
        "y" => vec![
            ("border-block-style", "var(--tw-border-style)".to_string()),
            ("border-block-width", "1px".to_string()),
        ],
        _ => return None,
    };
    Some((props, ()))
}

/// Padding/margin/gap utilities over the spacing scale. Returns `Ok(None)` when
/// the prefix is not a spacing family, `Err` when it is but the step is invalid.
fn spacing_utility(base: &str) -> Result<Option<Utility>, String> {
    let families: [(&str, &str); 10] = [
        ("gap-", "gap"),
        ("p-", "padding"),
        ("px-", "padding-inline"),
        ("py-", "padding-block"),
        ("pt-", "padding-top"),
        ("pr-", "padding-right"),
        ("pb-", "padding-bottom"),
        ("pl-", "padding-left"),
        ("m-", "margin"),
        ("mt-", "margin-top"),
    ];
    for (prefix, property) in families {
        if let Some(step) = base.strip_prefix(prefix) {
            let value =
                spacing_value(step).ok_or_else(|| unknown(base))?;
            return Ok(Some(Utility::simple(vec![(property, value)])));
        }
    }
    Ok(None)
}

/// The spacing-scale value for a numeric step, matching Tailwind's compiled
/// output: `0` -> `0`, `1` -> `var(--spacing)`, otherwise `calc(var(--spacing) * n)`.
fn spacing_value(step: &str) -> Option<String> {
    if step.is_empty() || !step.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some(match step {
        "0" => "0".to_string(),
        "1" => "var(--spacing)".to_string(),
        n => format!("calc(var(--spacing) * {n})"),
    })
}

/// Resolves a Tailwind color token (`black`, `white`, or `<hue>-<shade>`) to a
/// `var(--color-…)` reference, if the theme defines it.
fn resolve_color(token: &str) -> Option<String> {
    let var = format!("--color-{token}");
    if THEME_CSS.contains(&format!("{var}:")) || THEME_CSS.contains(&format!("{var} :")) {
        Some(format!("var({var})"))
    } else {
        None
    }
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

/// Escapes a class name for use in a selector: `:` and `/` become `\:` / `\/`.
fn escape_class(class: &str) -> String {
    let mut out = String::with_capacity(class.len());
    for c in class.chars() {
        if matches!(c, ':' | '/' | '.') {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// The app's own CSS with `@apply` expanded and the framework import removed.
struct UserCss {
    /// The `@layer base` body: the app's base rules (with `@apply` expanded) plus
    /// their `dark:` companions as `@media (prefers-color-scheme:dark)` rules.
    base_layer: String,
}

/// Processes the app CSS: strips `@import 'tailwindcss'`, walks each `@layer base`
/// block, and expands `@apply` directives into declarations (splitting `dark:`
/// applies into a dark media rule).
fn process_user_css(css: &str, tw_props: &mut BTreeSet<TwProp>) -> Result<UserCss, String> {
    let mut base_layer = String::new();
    let items = parse_top_level(css)?;
    for item in items {
        match item {
            TopItem::Import => {}
            TopItem::Layer { names, body } => {
                if names.split_whitespace().next() != Some("base") {
                    return Err(format!(
                        "unsupported at-rule `@layer {names}` in Tailwind CSS entry (native compiler handles `@layer base`)"
                    ));
                }
                let rules = parse_rules(&body)?;
                for rule in rules {
                    let (main, dark) = expand_rule(&rule, tw_props)?;
                    if let Some(main) = main {
                        base_layer.push_str(&main);
                    }
                    if let Some(dark) = dark {
                        base_layer.push_str("@media (prefers-color-scheme:dark){");
                        base_layer.push_str(&dark);
                        base_layer.push('}');
                    }
                }
            }
        }
    }
    Ok(UserCss { base_layer })
}

enum TopItem {
    Import,
    Layer { names: String, body: String },
}

/// Splits the entry into top-level items: `@import` statements and `@layer …{…}`
/// blocks. Errors on any other top-level construct.
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
        return Err(format!(
            "unsupported top-level CSS construct in Tailwind entry near: {:?}",
            &css[i..(i + 40).min(css.len())]
        ));
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
                let utility = generate_utility(apply_base, class, tw_props)?;
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
    }

    #[test]
    fn unknown_variant_is_a_hard_error() {
        let err = compile("@import 'tailwindcss';", &candidates(&["lg:flex"])).unwrap_err();
        assert!(err.contains("lg"), "error must name the variant: {err}");
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
