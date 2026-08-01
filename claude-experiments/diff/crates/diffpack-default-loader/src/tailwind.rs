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
//! heuristic): it extracts the string values that *flow into* class positions —
//! `class`/`className` attributes and `…Class`-suffixed props — through
//! literals, templates, ternaries (all three parts), call arguments,
//! `&&`/`||` chains, arrays and object literals, plus `const` bindings those
//! positions reference (resolved across all scanned files) and
//! `tailwind.config.*` `safelist` arrays. Compared string operands
//! (`phase !== 'finished'`) are excluded, so candidates stay real classes the
//! app applies and an unhandled one is a genuine gap. Candidates Tailwind
//! itself rejects (an `animate-*` with no theme token, a malformed variant)
//! are skipped exactly like the reference: it generates nothing for them
//! either.
//!
//! The theme defaults come from the app's own installed
//! `node_modules/tailwindcss/theme.css` when present (tokens like
//! `--font-sans` changed between v4 releases), resolved Node-style by walking up
//! from the STYLESHEET, and falling back to the vendored copy below.
//!
//! Color opacity modifiers (`bg-black/30`) compile to the modern
//! `color-mix(in oklab, …)` declaration Tailwind emits; the static sRGB fallback
//! hex the reference minifier additionally inlines for pre-`color-mix` browsers
//! is not duplicated (every target browser resolves the color-mix branch).
//!
//! The compile is a build-emit step (it runs once per `emit`, like manifest
//! generation), not part of the incremental transform hot path: a leaf edit still
//! re-transforms exactly one module.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use rayon::prelude::*;

/// The upstream `tailwindcss` release every vendored artifact below was taken
/// from. ONE definition: the banner is formatted from it, the installed-theme
/// resolver compares against it, and `tests/tailwind_upstream_drift.rs` asserts
/// it still matches the pin in `integration/tanstack-start-reference` and the
/// bytes of the vendored files. Re-vendoring is:
///
/// ```text
/// cp <tailwindcss>/theme.css     src/tailwind_theme.css
/// cp <tailwindcss>/preflight.css src/tailwind_preflight.source.css
/// ```
///
/// plus re-extracting the compiled preflight from a real reference build (see
/// [`PREFLIGHT_CSS`]) and bumping this constant.
pub const VERSION: &str = "4.3.3";

/// The published Tailwind default theme, verbatim from
/// `node_modules/tailwindcss/theme.css` at [`VERSION`]. Parsed for the token
/// values the app's utilities reference (colors, spacing, font sizes/weights,
/// radii, fonts).
const THEME_CSS: &str = include_str!("tailwind/theme.css");

/// The vendored default v4 theme (the `tailwindcss` package's `theme.css`). Exposed so
/// a legacy v3 app (which has no installed `tailwindcss/theme.css`) can start from this
/// base and merge its `tailwind.config.js` tokens on top, instead of the config-only
/// tokens replacing the whole default scale.
pub fn vendored_theme_css() -> &'static str {
    THEME_CSS
}

/// The resolved Tailwind preflight (base reset) at [`VERSION`]. Identical for
/// every v4 app — faithful reference data, not app-specific.
///
/// This is the COMPILED form: upstream ships `preflight.css` as commented
/// source, and the browser receives it after lightningcss has split
/// `::file-selector-button` into its own rules, rewritten `::after` to `:after`,
/// lowered `--theme(...)` to `var(...)` and synthesized a `color-mix` fallback
/// `@supports`. It therefore cannot be re-derived from
/// [`vendored_preflight_source_css`] in Rust; it is lifted verbatim out of a
/// real `@tailwindcss/vite` build, which is exactly what the drift guard's T5
/// asserts.
const PREFLIGHT_CSS: &str = include_str!("../../../src/tailwind_preflight.css");

/// Upstream's `preflight.css` SOURCE, verbatim at [`VERSION`]. Not used by the
/// compile (see [`PREFLIGHT_CSS`]) — it is the provenance record that lets the
/// drift guard notice an upstream preflight change without needing a rebuilt
/// reference.
const PREFLIGHT_SOURCE_CSS: &str = include_str!("tailwind/preflight.source.css");

/// The upstream Tailwind **v3** release [`PREFLIGHT_V3_SOURCE_CSS`] was vendored
/// from. Re-vendoring is `cp <tailwindcss@3>/src/css/preflight.css
/// src/tailwind_preflight_v3.source.css` plus bumping this.
pub const V3_VERSION: &str = "3.4.19";

/// Upstream's **v3** `preflight.css`, verbatim at [`V3_VERSION`].
///
/// A v3 app's base reset is NOT v4's. The user-visible divergence is the border
/// reset: v3 resets every element's `border-color` to `theme('borderColor.DEFAULT')`
/// (gray-200, `#e5e7eb`), v4 to `currentColor` — so compiling a v3 app with the v4
/// preflight recoloured every default border to the inherited text colour. v3's
/// `theme(...)` calls are resolved against the app's own resolved theme by
/// [`v3_preflight`] (this is unquestionably the *source* form: v3 emits it through
/// PostCSS with no lightningcss lowering, so unlike [`PREFLIGHT_CSS`] it needs no
/// separately extracted compiled copy).
const PREFLIGHT_V3_SOURCE_CSS: &str = include_str!("tailwind/preflight_v3.source.css");

/// The vendored v3 preflight source. Exposed for the drift guard.
pub fn vendored_preflight_v3_source_css() -> &'static str {
    PREFLIGHT_V3_SOURCE_CSS
}

/// The compiled preflight the engine emits. Exposed for the drift guard.
pub fn vendored_preflight_css() -> &'static str {
    PREFLIGHT_CSS
}

/// Upstream's preflight source the compiled form was derived from. Exposed for
/// the drift guard.
pub fn vendored_preflight_source_css() -> &'static str {
    PREFLIGHT_SOURCE_CSS
}

/// The banner Tailwind stamps on its output, carrying [`VERSION`].
fn version_banner() -> String {
    format!("/*! tailwindcss v{VERSION} | MIT License | https://tailwindcss.com */")
}

/// The `@supports` feature query Tailwind v4 guards its registered-property
/// fallbacks with (for browsers without `@property`).
const PROPERTIES_SUPPORTS: &str = "@supports (((-webkit-hyphens:none)) and (not (margin-trim:inline))) or ((-moz-orient:inline) and (not (color:rgb(from red r g b))))";

/// Which Tailwind dialect a stylesheet compiles as.
///
/// A legacy v3 entry (`@tailwind base;`) is not "v4 with a v3 theme": three
/// utility FORMS differ, and each one is visible in a browser's computed style.
///
/// * `box-shadow` composes 3 slots in v3, 5 in v4 (v4 added `inset-shadow` and
///   `inset-ring`), so `shadow-sm` computes to a 3- vs 5-layer `box-shadow`.
/// * `text-<size>` writes its line-height literally in v3; in v4 it writes
///   `var(--tw-leading, …)` so that a `leading-*` on the same element wins
///   regardless of source order. Compiling a v3 app the v4 way made
///   `md:text-4xl leading-tight` compute 45px where v3 computes 40px.
/// * `leading-*` therefore also sets `--tw-leading` in v4 but not in v3.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dialect {
    /// A legacy `@tailwind base; @tailwind utilities;` entry.
    V3,
    /// A v4 `@import "tailwindcss"` entry.
    V4,
}

impl Dialect {
    /// The dialect a Tailwind CSS entry point compiles as.
    pub fn of(css: &str) -> Self {
        if is_tailwind_v3_entry(css) {
            Self::V3
        } else {
            Self::V4
        }
    }

    /// The `box-shadow` composition every shadow/ring utility assigns.
    fn box_shadow_chain(self) -> &'static str {
        match self {
            Self::V3 => BOX_SHADOW_CHAIN_V3,
            Self::V4 => BOX_SHADOW_CHAIN,
        }
    }
}

/// The full `box-shadow` composition every shadow/ring utility assigns, verbatim
/// from Tailwind v4.
const BOX_SHADOW_CHAIN: &str = "var(--tw-inset-shadow), var(--tw-inset-ring-shadow), var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow)";

/// The same composition in Tailwind **v3**, verbatim: only three slots, and the
/// ring ones carry their own `0 0 #0000` fallback.
const BOX_SHADOW_CHAIN_V3: &str =
    "var(--tw-ring-offset-shadow, 0 0 #0000), var(--tw-ring-shadow, 0 0 #0000), var(--tw-shadow)";

/// Output rank of `text-<size>` under [`Dialect::V3`]. v3 registers its `fontSize`
/// plugin before `lineHeight`, so a `text-<size>`'s line-height loses to a
/// `leading-*` on the same element; every other utility keeps the default rank
/// (100), so a rank below it puts the sizes first exactly as v3 does.
const TEXT_SIZE_RANK_V3: u16 = 98;

/// The `filter` composition every filter utility assigns, verbatim from
/// Tailwind v4.
const FILTER_CHAIN: &str = "var(--tw-blur,) var(--tw-brightness,) var(--tw-contrast,) var(--tw-grayscale,) var(--tw-hue-rotate,) var(--tw-invert,) var(--tw-saturate,) var(--tw-sepia,) var(--tw-drop-shadow,)";

/// The `transform` composition the rotate/skew slots feed into, verbatim from
/// Tailwind v4. `transform`, `transform-cpu`, `rotate-x/y/z`, and `skew-*` all
/// assign this (with `transform-gpu` prepending `translateZ(0)`).
const TRANSFORM_CHAIN: &str = "var(--tw-rotate-x,) var(--tw-rotate-y,) var(--tw-rotate-z,) var(--tw-skew-x,) var(--tw-skew-y,)";

/// The `backdrop-filter` composition every backdrop utility assigns, verbatim
/// from Tailwind v4 (also emitted with the `-webkit-` prefix).
const BACKDROP_FILTER_CHAIN: &str = "var(--tw-backdrop-blur,) var(--tw-backdrop-brightness,) var(--tw-backdrop-contrast,) var(--tw-backdrop-grayscale,) var(--tw-backdrop-hue-rotate,) var(--tw-backdrop-invert,) var(--tw-backdrop-opacity,) var(--tw-backdrop-saturate,) var(--tw-backdrop-sepia,)";

/// `--tw-gradient-stops` as assigned by `from-*`/`to-*` color stops: the via
/// chain when a `via-*` is present, else position + from/to stops. Verbatim
/// from Tailwind v4.
const GRADIENT_STOPS: &str = "var(--tw-gradient-via-stops, var(--tw-gradient-position), var(--tw-gradient-from) var(--tw-gradient-from-position), var(--tw-gradient-to) var(--tw-gradient-to-position))";

/// `--tw-gradient-via-stops` as assigned by `via-*` color stops, verbatim from
/// Tailwind v4.
const GRADIENT_VIA_STOPS: &str = "var(--tw-gradient-position), var(--tw-gradient-from) var(--tw-gradient-from-position), var(--tw-gradient-via) var(--tw-gradient-via-position), var(--tw-gradient-to) var(--tw-gradient-to-position)";

/// Utility rules grouped for output: `(variant order, media key)` ->
/// `(family rank, class, rule css)` entries, sorted within each group.
type RuleGroups = BTreeMap<(u8, String), Vec<(u16, String, String)>>;

/// Why a candidate failed to compile.
///
/// `Unsupported` is an engine gap: a form Tailwind would generate but this
/// compiler does not yet — a hard error for recognized-root candidates, so a
/// missing style never ships silently. `Invalid` is a candidate Tailwind itself
/// rejects and generates nothing for (a value that resolves against no theme
/// token, like `animate-bounce-in` with no `--animate-bounce-in`, or a malformed
/// variant like `!dark:`): skipping it *is* reference parity.
enum Fail {
    Invalid,
    Unsupported(String),
}

impl Fail {
    /// The error for `@apply <class>`: there even a Tailwind-invalid class is a
    /// hard error, because the app's own stylesheet explicitly demands it.
    fn into_apply_error(self, class: &str) -> String {
        match self {
            Fail::Unsupported(msg) => msg,
            Fail::Invalid => format!(
                "`@apply {class}`: not a valid Tailwind utility (its value resolves against no theme token)"
            ),
        }
    }
}

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

/// Whether a CSS source is a legacy Tailwind **v3** entry point: it opts into the
/// framework via the three `@tailwind base|components|utilities` directives (a
/// `@tailwind screens|variants` also counts). The v4 compiler emits exactly the
/// `@layer base(+preflight)/components/utilities` cascade those directives expand to,
/// so a v3 entry compiles natively through the SAME path — the directives are consumed
/// as no-op markers in `parse_top_level`.
pub fn is_tailwind_v3_entry(css: &str) -> bool {
    css.lines().any(|line| {
        let rest = match line.trim().strip_prefix("@tailwind") {
            Some(rest) => rest.trim(),
            None => return false,
        };
        let keyword = rest.trim_end_matches(';').trim();
        matches!(
            keyword,
            "base" | "components" | "utilities" | "screens" | "variants"
        )
    })
}

/// Whether a CSS source should run through the native Tailwind compiler — a v4
/// (`@import "tailwindcss"`) OR a legacy v3 (`@tailwind …`) entry. This is the single
/// gate every capture/emit site uses so both dialects take the identical path.
pub fn needs_native_tailwind_compile(css: &str) -> bool {
    is_tailwind_entry(css) || is_tailwind_v3_entry(css)
}

// ---------------------------------------------------------------------------
// Native-engine capability check (see `crate::tailwind_delegate`)
// ---------------------------------------------------------------------------

/// Plain CSS at-rules a Tailwind entry may carry at top level that the compiler
/// reproduces verbatim: they hold no Tailwind semantics.
const VERBATIM_AT_RULES: &[&str] = &[
    "@keyframes",
    "@media",
    "@supports",
    "@font-face",
    "@property",
    "@container",
    "@counter-style",
    "@font-feature-values",
    "@page",
    "@scope",
    "@starting-style",
];

/// At-rules the native engine gives a Tailwind meaning to at top level.
const TAILWIND_AT_RULES: &[&str] = &[
    "@import",
    "@layer",
    "@source",
    "@config",
    "@tailwind",
    "@theme",
    "@custom-variant",
    "@utility",
];

/// Why the native Tailwind engine cannot serve a given entry.
///
/// Each variant is a *capability* gap, decided by reading the entry — never by
/// compiling it and catching a failure. A malformed stylesheet is NOT a gap: it
/// produces none of these and stays a hard error from the native compiler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeGap {
    /// `@plugin "<name>"` — a JavaScript Tailwind plugin. Its utilities,
    /// variants and base rules are arbitrary JS; no CSS-level engine can know
    /// them.
    Plugin(String),
    /// A top-level at-rule with no native meaning (`@variant`, `@reference`, …).
    AtRule(String),
    /// `@apply <class>` naming a utility neither a built-in nor an app
    /// `@utility` provides — in practice, one a `@plugin` registers.
    Apply { class: String, detail: String },
}

impl std::fmt::Display for NativeGap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NativeGap::Plugin(name) => write!(
                f,
                "it loads the JavaScript Tailwind plugin `{name}` (`@plugin`), whose utilities \
                 and variants exist only inside that plugin"
            ),
            NativeGap::AtRule(rule) => write!(
                f,
                "it uses the at-rule `{rule}`, which diffpack's Tailwind engine does not implement"
            ),
            NativeGap::Apply { class, detail } => write!(
                f,
                "`@apply {class}` names a utility no built-in and no `@utility` provides ({detail})"
            ),
        }
    }
}

/// Reads a Tailwind entry and reports the first thing in it the native engine
/// cannot serve, or `None` when the native engine owns the whole sheet.
///
/// This is the delegation gate for [`crate::tailwind_delegate`]. It is a
/// *capability* question answered from the CSS: the three gaps in [`NativeGap`]
/// are enumerated and looked for directly. Anything else — malformed CSS, an
/// unresolvable theme reference, a scanned class that is simply not a utility —
/// is not a gap and keeps its existing native behavior (hard error, or the
/// lenient skip [`compile_with_theme_lenient`] performs).
///
/// `app_theme_css` is the same theme source the native compile would use; the
/// `@apply` probe needs it because whether `text-brand` resolves depends on the
/// tokens in scope.
pub fn native_gap(css: &str, app_theme_css: Option<&str>) -> Option<NativeGap> {
    if let Some(gap) = at_rule_gap(css) {
        return Some(gap);
    }
    apply_gap(css, app_theme_css)
}

/// Scans the entry's top-level at-rules for `@plugin` or for a rule outside the
/// native set. Strings, comments and nested (in-block) at-rules are skipped, so
/// a `@media` body's `@apply` or a `content: "@plugin"` never matches.
fn at_rule_gap(css: &str) -> Option<NativeGap> {
    let bytes = css.as_bytes();
    let mut depth = 0usize;
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'"' | b'\'' => {
                let quote = bytes[index];
                index += 1;
                while index < bytes.len() && bytes[index] != quote {
                    index += if bytes[index] == b'\\' { 2 } else { 1 };
                }
                index += 1;
            }
            b'/' if bytes.get(index + 1) == Some(&b'*') => {
                index = match css[index + 2..].find("*/") {
                    Some(rel) => index + 2 + rel + 2,
                    None => bytes.len(),
                };
            }
            b'{' => {
                depth += 1;
                index += 1;
            }
            b'}' => {
                depth = depth.saturating_sub(1);
                index += 1;
            }
            b'@' if depth == 0 => {
                let mut end = index + 1;
                while end < bytes.len()
                    && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'-')
                {
                    end += 1;
                }
                let name = &css[index..end];
                if name == "@plugin" {
                    return Some(NativeGap::Plugin(at_rule_argument(&css[end..])));
                }
                if !TAILWIND_AT_RULES.contains(&name) && !VERBATIM_AT_RULES.contains(&name) {
                    return Some(NativeGap::AtRule(name.to_string()));
                }
                index = end;
            }
            _ => index += 1,
        }
    }
    None
}

/// The quoted argument of an at-rule (`@plugin "tailwind-scrollbar" { … }` ->
/// `tailwind-scrollbar`), or the raw prelude when it carries no quotes.
fn at_rule_argument(rest: &str) -> String {
    let prelude = rest.split(['{', ';']).next().unwrap_or(rest).trim();
    match prelude.find(['"', '\'']) {
        Some(open) => {
            let quote = prelude.as_bytes()[open] as char;
            let inner = &prelude[open + 1..];
            match inner.find(quote) {
                Some(close) => inner[..close].to_string(),
                None => prelude.to_string(),
            }
        }
        None => prelude.to_string(),
    }
}

/// Resolves every `@apply` the entry's own rules make, against the same theme,
/// `@utility` and `@custom-variant` registry the native compile would build, and
/// reports the first one that has no answer.
///
/// Only the *resolution* runs — no candidate is rendered and no stylesheet is
/// assembled. A parse failure here is not a capability gap: it means the CSS is
/// malformed, which the native compile reports properly, so it yields `None`.
fn apply_gap(css: &str, app_theme_css: Option<&str>) -> Option<NativeGap> {
    let mut theme_src = app_theme_css.unwrap_or(THEME_CSS).to_string();
    let inline_theme = extract_theme_blocks(css);
    if !inline_theme.is_empty() {
        theme_src.push('\n');
        theme_src.push_str(&inline_theme);
    }
    Theme::validate_wildcards(&theme_src).ok()?;
    let theme = Theme::parse(&theme_src);
    let mut custom_variants = scan_custom_variants(&theme_src).ok()?;
    let items = parse_top_level(css).ok()?;
    let mut custom_utilities = CustomUtilities::default();
    for item in &items {
        match item {
            TopItem::CustomVariant { name, template } => {
                custom_variants.insert(name.clone(), template.clone());
            }
            TopItem::Utility { name, body } => match name.strip_suffix("-*") {
                Some(prefix) if !prefix.is_empty() && !prefix.contains('*') => {
                    custom_utilities
                        .functional
                        .insert(prefix.to_string(), body.clone());
                }
                Some(_) => return None,
                None if !name.contains('*') => {
                    custom_utilities.statics.insert(name.clone(), body.clone());
                }
                None => return None,
            },
            _ => {}
        }
    }
    let dialect = Dialect::of(css);
    // Scratch: the probe generates nothing, so the `--tw-*` property registrations
    // an expansion would contribute are discarded.
    let mut tw_props = BTreeSet::new();
    let mut probe = |rule: &StyleRule| -> Option<NativeGap> {
        let error = expand_rule(rule, &theme, &mut tw_props, &custom_utilities, dialect).err()?;
        Some(NativeGap::Apply {
            class: applied_class(&error).unwrap_or_else(|| rule.selector.clone()),
            detail: error,
        })
    };
    for item in items {
        match item {
            TopItem::Layer { names, body } => {
                if !matches!(
                    names.split_whitespace().next(),
                    Some("base" | "components" | "utilities")
                ) {
                    continue;
                }
                for rule in parse_rules(&body).ok()? {
                    if let Some(gap) = probe(&rule) {
                        return Some(gap);
                    }
                }
            }
            TopItem::Rule { selector, body } => {
                if let Some(gap) = probe(&StyleRule { selector, body }) {
                    return Some(gap);
                }
            }
            _ => {}
        }
    }
    None
}

/// The class named by an `@apply` expansion error (`` `@apply foo`: … ``).
fn applied_class(error: &str) -> Option<String> {
    let rest = error.strip_prefix("`@apply ")?;
    let end = rest.find('`')?;
    Some(rest[..end].to_string())
}

/// Compiles a Tailwind v4 CSS entry into a plain, self-contained stylesheet.
///
/// `candidate_classes` are the utility class tokens scanned from the app's source
/// (see [`scan_class_candidates`]). Every candidate must resolve to a utility —
/// or to a class the app's own CSS defines — or this returns a hard error naming
/// every unresolved token.
pub fn compile(css: &str, candidate_classes: &BTreeSet<String>) -> Result<String, String> {
    compile_impl(css, candidate_classes, None, true)
}

/// Like [`compile_with_theme`] but LENIENT: an unresolved candidate — a recognized
/// utility root the engine hasn't implemented, OR a legacy/removed class the app
/// still references (`bg-opacity-90`) that Tailwind itself rejects — is warned about
/// on stderr and skipped rather than being a hard error. This matches Tailwind's own
/// scanner, which silently ignores every non-utility token it finds in the source, so
/// a real app (whose scanned classes inevitably include such tokens) still builds.
/// Used for real-app builds; the conformance path stays strict via [`compile`].
pub fn compile_with_theme_lenient(
    css: &str,
    candidate_classes: &BTreeSet<String>,
    app_theme_css: Option<&str>,
) -> Result<String, String> {
    compile_impl(css, candidate_classes, app_theme_css, false)
}

/// [`compile`] against an app-provided theme source — the app's own installed
/// `node_modules/tailwindcss/theme.css` when present, so the compile matches
/// the exact Tailwind version the reference build used (default tokens like
/// `--font-sans` changed between v4 releases). Falls back to the vendored
/// [`THEME_CSS`].
pub fn compile_with_theme(
    css: &str,
    candidate_classes: &BTreeSet<String>,
    app_theme_css: Option<&str>,
) -> Result<String, String> {
    compile_impl(css, candidate_classes, app_theme_css, true)
}

fn compile_impl(
    css: &str,
    candidate_classes: &BTreeSet<String>,
    app_theme_css: Option<&str>,
    strict: bool,
) -> Result<String, String> {
    // The base theme (the app's installed `tailwindcss/theme.css` if found, else the
    // embedded default) EXTENDED with any inline `@theme { … }` blocks the app's own
    // stylesheet declares. A later declaration overrides an earlier one, so the app's
    // tokens win — this is what makes create-next-app's default `@theme inline`
    // (`--font-sans`, `--color-background`, …) resolve for `font-sans`, `bg-*`, etc.
    let mut theme_src = app_theme_css.unwrap_or(THEME_CSS).to_string();
    let app_theme = extract_theme_blocks(css);
    if !app_theme.is_empty() {
        theme_src.push('\n');
        theme_src.push_str(&app_theme);
    }
    Theme::validate_wildcards(&theme_src)?;
    let theme = Theme::parse(&theme_src);
    let mut tw_props: BTreeSet<TwProp> = BTreeSet::new();
    let dialect = Dialect::of(css);

    // A legacy JS config's `darkMode: 'class' | 'selector' | [...]` reaches the
    // compiler as a `@custom-variant dark (…)` line in the config-derived theme
    // source (`scripts/tailwind-config-eval.mjs` emits it), because that is exactly
    // what the option means: `dark:` is a SELECTOR variant, not a media query.
    // Ignoring it compiled every `dark:` utility into
    // `@media (prefers-color-scheme: dark)`, so a class-toggled app rendered its
    // dark palette purely because the browser preferred dark.
    let theme_variants = scan_custom_variants(&theme_src)?;

    // 1. Process the app's own CSS first: strip the framework import, expand
    //    `@apply` inside `@layer base`, pass plain (unlayered) rules through,
    //    and learn which class names the app's own CSS defines.
    let user = process_user_css(css, &theme, &mut tw_props, theme_variants, dialect)?;

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
        match render_utility(
            class,
            &theme,
            &mut tw_props,
            &user.custom_variants,
            &user.custom_utilities,
            dialect,
        ) {
            Ok(rule) => groups
                .entry((rule.order, rule.media_key))
                .or_default()
                .push((rule.rank, class.clone(), rule.css)),
            // Tailwind's own scanner treats EVERY string token in the source
            // tree as a candidate and silently ignores the ones that are not
            // utilities at all (`zero`, `data`, ...). Matching that: a token
            // whose root no utility family recognizes is skipped, and a token
            // Tailwind itself rejects (`Fail::Invalid`) generates nothing in
            // the reference either. A token with a RECOGNIZED root whose form
            // the engine has not implemented stays a hard error — that is the
            // surface where silence would ship a broken style.
            Err(Fail::Invalid) => {}
            Err(Fail::Unsupported(error)) => {
                if utility_root_recognized(class, &user.custom_utilities) {
                    // strict: keep the full diagnostic (a real engine gap to fix);
                    // lenient: just the class name for a compact skip warning.
                    errors.push(if strict { error } else { class.clone() });
                }
            }
        }
    }
    if !errors.is_empty() {
        if strict {
            return Err(errors.join("\n"));
        }
        // Lenient: match Tailwind's scanner — skip the unresolved candidates, but say
        // so loudly (never silently), so a real engine gap stays visible and fixable.
        errors.sort();
        eprintln!(
            "[tailwind] {} scanned class(es) not generated (skipped, as Tailwind's scanner does — some may be legacy/removed utilities the app still references): {}",
            errors.len(),
            errors.join(", ")
        );
    }

    // 3. Determine which theme tokens the generated CSS references.
    let mut referenced: BTreeSet<String> = BTreeSet::new();
    for rules in groups.values() {
        for (_, _, css) in rules {
            collect_theme_vars_str(css, &theme, &mut referenced);
        }
    }
    collect_theme_vars_str(&user.base_layer, &theme, &mut referenced);
    collect_theme_vars_str(&user.components_layer, &theme, &mut referenced);
    collect_theme_vars_str(&user.utilities_layer, &theme, &mut referenced);
    collect_theme_vars_str(&user.postlude, &theme, &mut referenced);
    // The preflight always relies on the default font-family tokens.
    for always in [
        "--font-sans",
        "--font-mono",
        "--default-font-family",
        "--default-mono-font-family",
    ] {
        if theme.contains(always) {
            referenced.insert(always.to_string());
        }
    }

    // 4. Assemble the stylesheet, layer by layer, matching Tailwind v4 order.
    let mut out = String::new();
    out.push_str(&version_banner());
    out.push('\n');

    // A few properties are *unregistered* under the v3 dialect: v3 declares them
    // in a plain `*, ::before, ::after` defaults rule with an EMPTY value, which
    // `@property` cannot express (a registered `syntax: "*"` property with no
    // initial value is guaranteed-invalid, so `var()`-ing it would poison the
    // whole declaration). They are emitted separately, below.
    let (empty_defaults, tw_props): (BTreeSet<TwProp>, BTreeSet<TwProp>) = tw_props
        .into_iter()
        .partition(|prop| prop.empty_default_in(dialect));

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

    // A legacy v3 entry (`@tailwind base`) gets the v3 base reset, not v4's: the two
    // differ user-visibly (v3 resets `border-color` to gray-200, v4 to `currentColor`).
    let preflight = if dialect == Dialect::V3 {
        v3_preflight(&theme)?
    } else {
        PREFLIGHT_CSS.to_string()
    };
    out.push_str("@layer base{");
    out.push_str(&preflight);
    if !empty_defaults.is_empty() {
        // v3's own defaults rule, verbatim in shape: `--tw-…: ;` (an empty value,
        // so `var(--tw-…)` substitutes nothing) on the universal selector, right
        // after the preflight.
        out.push_str("*,::before,::after,::backdrop{");
        out.push_str(
            &empty_defaults
                .iter()
                .map(|prop| format!("{}: ", prop.spec().0))
                .collect::<Vec<_>>()
                .join(";"),
        );
        out.push('}');
    }
    out.push_str(&user.base_layer);
    out.push('}');

    // The components layer: empty (a bare layer declaration that just fixes the
    // layer's cascade position) unless the app wrote `@layer components { … }`.
    if user.components_layer.is_empty() {
        out.push_str("@layer components;");
    } else {
        out.push_str("@layer components{");
        out.push_str(&user.components_layer);
        out.push('}');
    }

    out.push_str("@layer utilities{");
    for ((_, media_key), mut rules) in groups {
        rules.sort();
        let conditions: Vec<&str> = media_key.split('|').filter(|c| !c.is_empty()).collect();
        for condition in &conditions {
            // A condition already prefixed with `@` is a bare at-rule wrapper
            // (e.g. `@starting-style`); emit it verbatim. Everything else is a
            // media-feature/range condition wrapped in `@media (...)`.
            if let Some(at_rule) = condition.strip_prefix('@') {
                out.push('@');
                out.push_str(at_rule);
                out.push('{');
            } else {
                out.push_str("@media ");
                out.push_str(&format_media_query(condition));
                out.push('{');
            }
        }
        for (_, _, css) in &rules {
            out.push_str(css);
        }
        for _ in &conditions {
            out.push('}');
        }
    }
    // The app's own `@layer utilities` rules follow the generated ones, which is
    // where they sit relative to its `@import 'tailwindcss'`.
    out.push_str(&user.utilities_layer);
    out.push('}');

    // The app's own plain (unlayered) rules follow the layers, exactly where
    // they sit relative to the `@import 'tailwindcss'` in the source.
    out.push_str(&user.postlude);

    for prop in &tw_props {
        out.push_str(&prop.property_declaration());
    }

    // @keyframes for every referenced --animate-* token, emitted last like the
    // reference build (the app's own keyframes already live in its plain CSS).
    let mut emitted_keyframes: BTreeSet<&str> = BTreeSet::new();
    for name in &referenced {
        if !name.starts_with("--animate-") {
            continue;
        }
        let Some(value) = theme.get(name) else {
            continue;
        };
        let animation = value.split_whitespace().next().unwrap_or("");
        if let Some(body) = theme.keyframes(animation)
            && emitted_keyframes.insert(animation)
        {
            out.push_str(body);
        }
    }

    Ok(out)
}

/// Renders the Tailwind **v3** preflight for `theme`: the vendored v3 source with
/// its comments stripped and every `theme('<path>', <fallback>)` call resolved
/// against the app's resolved theme (falling back to the literal upstream wrote).
///
/// An unrecognized `theme()` path is a HARD ERROR naming it, not a silent
/// pass-through: the vendored file is fixed, so a path this does not map can only
/// mean the vendored preflight moved ahead of this resolver, and shipping the
/// literal `theme(...)` text would put an invalid declaration in the base layer.
fn v3_preflight(theme: &Theme) -> Result<String, String> {
    let stripped = strip_css_comments(PREFLIGHT_V3_SOURCE_CSS);
    let mut out = String::with_capacity(stripped.len());
    let mut rest = stripped.as_str();
    while let Some(at) = rest.find("theme(") {
        out.push_str(&rest[..at]);
        let after = &rest[at + "theme(".len()..];
        let close = after.find(')').ok_or_else(|| {
            format!(
                "tailwind v3 preflight: unterminated theme() call at `{}`",
                &after[..40.min(after.len())]
            )
        })?;
        let args = &after[..close];
        let (path, fallback) = match args.split_once(',') {
            Some((path, fallback)) => (path.trim(), fallback.trim()),
            None => (args.trim(), ""),
        };
        let path = path.trim_matches(['\'', '"']);
        let var = v3_theme_path_var(path).ok_or_else(|| {
            format!(
                "tailwind v3 preflight: no theme mapping for `theme('{path}')` — \
                 src/tailwind_preflight_v3.source.css has moved ahead of v3_theme_path_var()"
            )
        })?;
        out.push_str(theme.get(&var).unwrap_or(fallback));
        rest = &after[close + 1..];
    }
    out.push_str(rest);
    Ok(compact_css(&out))
}

/// The v4 theme variable a v3 `theme('<path>')` lookup resolves to, for the paths
/// the vendored v3 preflight uses. `None` means unmapped (a hard error upstream).
fn v3_theme_path_var(path: &str) -> Option<String> {
    // `fontFamily.sans[1].fontFeatureSettings` -> the `--font-sans--font-feature-settings`
    // modifier token the config evaluator emits alongside `--font-sans`.
    if let Some((head, modifier)) = path.split_once("[1].") {
        let base = v3_theme_path_var(head)?;
        return Some(format!("{base}--{}", kebab_case(modifier)));
    }
    let (category, rest) = path.split_once('.')?;
    match category {
        // v3's `borderColor.DEFAULT` has no v4 namespace; the config evaluator
        // publishes it as `--default-border-color`.
        "borderColor" if rest == "DEFAULT" => Some("--default-border-color".to_string()),
        "colors" => Some(format!("--color-{}", rest.replace('.', "-"))),
        "fontFamily" => Some(format!("--font-{rest}")),
        _ => None,
    }
}

/// `fontFeatureSettings` -> `font-feature-settings`.
fn kebab_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for c in s.chars() {
        if c.is_ascii_uppercase() {
            out.push('-');
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

/// Removes `/* … */` comments from a CSS source.
fn strip_css_comments(css: &str) -> String {
    let mut out = String::with_capacity(css.len());
    let mut rest = css;
    while let Some(at) = rest.find("/*") {
        out.push_str(&rest[..at]);
        match rest[at + 2..].find("*/") {
            Some(end) => rest = &rest[at + 2 + end + 2..],
            None => return out,
        }
    }
    out.push_str(rest);
    out
}

/// `group`/`peer` (optionally named, `group/name`) are variant marker classes:
/// Tailwind generates no CSS for them.
fn is_marker_class(class: &str) -> bool {
    class == "group" || class == "peer" || class.starts_with("group/") || class.starts_with("peer/")
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
    scan_class_candidates_multi(std::slice::from_ref(&source), out);
}

/// One file's contribution to the candidate scan, kept so a scan can be updated for
/// a single changed file instead of re-read and re-tokenized whole (the dev loop does
/// exactly that; see `Bundler::refresh_tailwind_scan_path`).
///
/// The three pieces are what the cross-file passes consume: the classes this file
/// states outright, the identifiers its class positions referenced, and the string
/// bindings it declares for those identifiers to resolve against.
#[derive(Clone)]
pub struct SourceScan {
    found: BTreeSet<String>,
    idents: BTreeSet<String>,
    bindings: Vec<(String, String)>,
}

/// Scan ONE file into its cacheable parts. Pass 1 of
/// [`scan_class_candidates_multi`], for one file, plus that file's binding
/// declarations.
pub fn scan_source_parts(source: &str) -> SourceScan {
    let mut found = BTreeSet::new();
    let mut idents = BTreeSet::new();
    scan_class_positions(source, &mut found, &mut idents);
    scan_safelist_arrays(source, &mut found, &mut idents);
    scan_class_helper_calls(source, &mut found, &mut idents);
    SourceScan {
        found,
        idents,
        bindings: binding_initializers(source)
            .into_iter()
            .map(|(name, init)| (name.to_string(), init.to_string()))
            .collect(),
    }
}

/// The candidate set a group of scanned files produces: their stated classes, plus
/// every identifier their class positions referenced resolved against `const`/`let`/
/// `var` string bindings declared in ANY of them (`import { COLOR } from './colors'` +
/// `className={COLOR[kind]}`), iterated to a fixpoint so chains across files resolve.
///
/// This is the CROSS-FILE half of the scan, and it is why a scan cannot be cached as a
/// per-file union: a class only one file mentions can be reachable only through another
/// file's binding. Keeping the per-file parts and re-running this over them is both
/// exact and cheap — it re-reads nothing and re-tokenizes nothing.
pub fn resolve_scans<'a>(scans: impl Iterator<Item = &'a SourceScan>, out: &mut BTreeSet<String>) {
    let scans = scans.collect::<Vec<_>>();
    let mut idents: BTreeSet<String> = BTreeSet::new();
    for scan in &scans {
        out.extend(scan.found.iter().cloned());
        idents.extend(scan.idents.iter().cloned());
    }
    // Driven by an INDEX built in one pass over every file, not by re-scanning every
    // file for every identifier. Both find exactly the same bindings — the index is
    // the same predicate read from the declaration keyword instead of from the name —
    // but the rescan is O(identifiers x files) full-text searches, which on a
    // monorepo-sized source set (cal.com: thousands of files, hundreds of referenced
    // identifiers) was the single largest phase of a production build.
    let index_stage = diffpack_core::build_profile::stage("css/tailwind-binding-index");
    let mut bindings: HashMap<&str, Vec<&str>> = HashMap::new();
    for scan in &scans {
        for (name, init) in &scan.bindings {
            bindings
                .entry(name.as_str())
                .or_default()
                .push(init.as_str());
        }
    }
    drop(index_stage);
    resolve_idents(idents, &bindings, out);
}

/// Multi-file variant of [`scan_class_candidates`]: identifiers referenced
/// from one file's class positions resolve against `const` bindings in ANY of
/// the files (`import { COLOR } from './colors'` + `className={COLOR[kind]}`),
/// iterated to a fixpoint so chains across files resolve too.
pub fn scan_class_candidates_multi<S: AsRef<str> + Sync>(
    sources: &[S],
    out: &mut BTreeSet<String>,
) {
    // Pass 1: every file's class-valued positions and binding declarations.
    // Per-file independent, so it fans out across the pool.
    let positions_stage = diffpack_core::build_profile::stage("css/tailwind-scan-positions");
    let scanned = sources
        .par_iter()
        .map(|source| scan_source_parts(source.as_ref()))
        .collect::<Vec<_>>();
    drop(positions_stage);
    // Pass 2: the cross-file resolve, shared verbatim with the incremental path.
    resolve_scans(scanned.iter(), out);
}

/// Pass 2's fixpoint: every referenced identifier resolved against the binding index.
fn resolve_idents(
    idents: BTreeSet<String>,
    bindings: &HashMap<&str, Vec<&str>>,
    out: &mut BTreeSet<String>,
) {
    let _resolve_stage = diffpack_core::build_profile::stage("css/tailwind-resolve-idents");
    let mut visited: BTreeSet<String> = BTreeSet::new();
    let mut worklist: Vec<String> = idents.into_iter().collect();
    while let Some(name) = worklist.pop() {
        if !visited.insert(name.clone()) {
            continue;
        }
        let Some(initializers) = bindings.get(name.as_str()) else {
            continue;
        };
        for init in initializers {
            let init = init.trim();
            // String-shaped: literals, templates, parenthesized
            // expressions, ternaries, and string-container literals
            // (`[…].join(' ')`, `{ primary: '…' }` maps indexed from a
            // class position).
            let eligible =
                init.starts_with(['"', '\'', '`', '(', '[', '{']) || split_ternary(init).is_some();
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

/// Collects the string entries of any `safelist: [...]` array — Tailwind's own
/// escape hatch for classes built dynamically (`grid-cols-${n}`), declared in
/// `tailwind.config.*`, which the reference scanner picks up like any other
/// source file.
fn scan_safelist_arrays(source: &str, out: &mut BTreeSet<String>, idents: &mut BTreeSet<String>) {
    let bytes = source.as_bytes();
    let mut i = 0;
    while let Some(rel) = source[i..].find("safelist") {
        let start = i + rel;
        i = start + "safelist".len();
        if start > 0 && is_ident_byte(bytes[start - 1]) {
            continue;
        }
        let mut j = skip_ws(bytes, i);
        if j >= bytes.len() || bytes[j] != b':' {
            continue;
        }
        j = skip_ws(bytes, j + 1);
        if j >= bytes.len() || bytes[j] != b'[' {
            continue;
        }
        let Some(end) = find_balanced(source, j, b'[', b']') else {
            continue;
        };
        collect_class_expression(&source[j + 1..end], out, idents);
        i = end + 1;
    }
}

/// The class-composition helpers whose ARGUMENTS are class positions by
/// definition. Calling one of these is the idiomatic way to build a class string,
/// and the call routinely sits somewhere no class-valued position scan reaches —
/// cal.com's embed button is `className = classNames("hidden lg:inline-flex",
/// className)`, a reassignment of a destructured parameter, so neither the JSX
/// attribute (it holds a bare identifier) nor the binding index (there is no
/// declaration) leads back to the literal. `lg:inline-flex` was therefore the one
/// utility in the whole app the reference build emitted and this one did not, and
/// the button rendered `hidden` at every viewport.
///
/// Tailwind's own scanner has no such gap because it is a raw text scan: every
/// candidate-shaped token in the file is a candidate, wherever it sits. Matching
/// that wholesale would trade this class of miss for a large over-generation, so
/// the narrower rule is the one Tailwind codebases actually rely on — these
/// helpers' arguments — and it costs nothing to be wrong about a call that merely
/// shares a name, because compilation is lenient: a candidate that resolves to no
/// utility is skipped, not emitted and not an error.
const CLASS_COMPOSITION_HELPERS: [&str; 9] = [
    "classNames",
    "classnames",
    "clsx",
    "cn",
    "cx",
    "twMerge",
    "twJoin",
    "cva",
    "tv",
];

/// Collects candidates from the arguments of every [`CLASS_COMPOSITION_HELPERS`]
/// call in `source`, wherever it appears.
fn scan_class_helper_calls(
    source: &str,
    out: &mut BTreeSet<String>,
    idents: &mut BTreeSet<String>,
) {
    let bytes = source.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if !is_ident_byte(bytes[i]) || bytes[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        // A whole identifier token: the byte before it must not be one, or this is
        // the tail of a longer name (`myCn`) and not a call to the helper.
        if i > 0 && is_ident_byte(bytes[i - 1]) {
            while i < bytes.len() && is_ident_byte(bytes[i]) {
                i += 1;
            }
            continue;
        }
        let mut j = i;
        while j < bytes.len() && is_ident_byte(bytes[j]) {
            j += 1;
        }
        let name = &source[i..j];
        let after = skip_ws(bytes, j);
        if CLASS_COMPOSITION_HELPERS.contains(&name)
            && after < bytes.len()
            && bytes[after] == b'('
            && let Some(end) = find_balanced(source, after, b'(', b')')
        {
            collect_class_expression(&source[after + 1..end], out, idents);
            i = end + 1;
            continue;
        }
        i = j;
    }
}

/// Walks class-valued positions — `class`/`className` attributes plus any
/// identifier ending in `Class`/`ClassName` (`btnClass={…}`, `divClass: '…'`,
/// the conventional names for class-carrying props) — and collects tokens and
/// referenced identifiers from their value expressions.
fn scan_class_positions(source: &str, out: &mut BTreeSet<String>, idents: &mut BTreeSet<String>) {
    let bytes = source.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let lower = source[i..].find("class");
        let upper = source[i..].find("Class");
        let rel = match (lower, upper) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => break,
        };
        let start = i + rel;
        let suffix_form = bytes[start] == b'C';
        if suffix_form {
            // `Class` must end a longer identifier (`btnClass`); a bare
            // `Class` token is not a class position.
            if start == 0 || !is_ident_byte(bytes[start - 1]) {
                i = start + 5;
                continue;
            }
        } else if start > 0 && is_ident_byte(bytes[start - 1]) {
            // Lowercase `class` must begin the identifier (`class`,
            // `className`), not sit inside one.
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
    if let Some((cond_part, then_part, else_part)) = split_ternary(expr) {
        collect_class_expression(cond_part, out, idents);
        collect_class_expression(then_part, out, idents);
        collect_class_expression(else_part, out, idents);
        return;
    }
    let bytes = expr.as_bytes();
    let mut i = 0;
    // Set right after a comparison operator: that operand is being compared
    // against (`phase !== 'finished'`), not applied as a class.
    let mut comparison_operand = false;
    while i < bytes.len() {
        i = skip_ws(bytes, i);
        if i >= bytes.len() {
            return;
        }
        match bytes[i] {
            b'"' | b'\'' | b'`' => {
                let end = if comparison_operand {
                    // The compared string is not a class list: walk past it
                    // (still resolving interpolations for identifiers).
                    let mut sink = BTreeSet::new();
                    collect_string_shape(expr, i, &mut sink, idents)
                } else {
                    collect_string_shape(expr, i, out, idents)
                };
                let Some(end) = end else { return };
                i = end;
                comparison_operand = false;
            }
            b'(' | b'[' => {
                let close = if bytes[i] == b'(' { b')' } else { b']' };
                let Some(end) = find_balanced(expr, i, bytes[i], close) else {
                    return;
                };
                collect_class_expression(&expr[i + 1..end], out, idents);
                i = end + 1;
                comparison_operand = false;
            }
            b'{' => {
                // Object literals: values (and classnames-style keys,
                // `{ 'is-active': cond }`) are class-shaped positions.
                let Some(end) = find_balanced(expr, i, b'{', b'}') else {
                    return;
                };
                collect_class_expression(&expr[i + 1..end], out, idents);
                i = end + 1;
                comparison_operand = false;
            }
            b'=' | b'!' if bytes.get(i + 1) == Some(&b'=') => {
                // `==`, `===`, `!=`, `!==`.
                i += 2;
                if bytes.get(i) == Some(&b'=') {
                    i += 1;
                }
                comparison_operand = true;
            }
            b'&' | b'|' if bytes.get(i + 1) == Some(&bytes[i]) => {
                // `&&` / `||`: the next operand is applied again.
                i += 2;
                comparison_operand = false;
            }
            b',' | b'+' => {
                i += 1;
                comparison_operand = false;
            }
            b'.' => {
                // Member access: skip the property name.
                i += 1;
                while i < bytes.len() && is_ident_byte(bytes[i]) {
                    i += 1;
                }
            }
            b if is_ident_byte(b) && !b.is_ascii_digit() => {
                let mut j = i;
                while j < bytes.len() && is_ident_byte(bytes[j]) {
                    j += 1;
                }
                let after = skip_ws(bytes, j);
                // A plain identifier reference contributes its binding's
                // strings; a call target does not (its arguments do, via the
                // parenthesis branch), and neither does a compared operand.
                if !comparison_operand && (after >= bytes.len() || bytes[after] != b'(') {
                    idents.insert(expr[i..j].to_string());
                }
                i = j;
                comparison_operand = false;
            }
            _ => {
                // Numbers, `!`, other operators: never class-shaped.
                i += 1;
            }
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

/// Finds every `const|let|var <name> = <initializer>` in the source and returns
/// the initializer texts (up to the top-level `;`).
/// Every `const`/`let`/`var NAME = <initializer>` binding in one source, as
/// `(name, initializer)` slices of it.
///
/// Driven from the declaration keyword, so ONE pass over the file yields the bindings
/// for every name at once. A destructuring pattern (`const { a } = x`) is not a name
/// binding and is skipped, as is `constFoo` (the keyword must end at a word boundary)
/// and `x == y` / `f = () =>` (the `=` must be a single assignment).
///
/// This is a deliberately naive text scan, exactly like the class-position scan beside
/// it: string and comment contents are not excluded. A candidate that only ever appears
/// inside a string is harmless (Tailwind's own scanner is likewise text-based and
/// over-collects), whereas parsing every file of a monorepo to be precise would cost
/// far more than the utilities it would exclude.
fn binding_initializers(source: &str) -> Vec<(&str, &str)> {
    const KEYWORDS: [&str; 3] = ["const", "let", "var"];
    let bytes = source.as_bytes();
    let mut found = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        // The next declaration keyword at or after `i`.
        let Some((start, keyword)) = KEYWORDS
            .iter()
            .filter_map(|kw| source[i..].find(kw).map(|rel| (i + rel, *kw)))
            .min_by_key(|(at, _)| *at)
        else {
            break;
        };
        let after_keyword = start + keyword.len();
        i = after_keyword;
        // The keyword must be a whole word: not the tail of `myconst`, not the head
        // of `constant`.
        if start > 0 && is_ident_byte(bytes[start - 1]) {
            continue;
        }
        if after_keyword < bytes.len() && is_ident_byte(bytes[after_keyword]) {
            continue;
        }
        // The bound NAME, over whitespace. Anything else (`{`, `[`) is a destructuring
        // pattern, which binds no single name this scan can resolve.
        let name_start = skip_ws(bytes, after_keyword);
        let mut name_end = name_start;
        while name_end < bytes.len() && is_ident_byte(bytes[name_end]) {
            name_end += 1;
        }
        if name_end == name_start {
            continue;
        }
        // Followed (over whitespace) by a single `=`.
        let mut k = skip_ws(bytes, name_end);
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
        found.push((
            &source[name_start..name_end],
            &source[init_start..k.min(bytes.len())],
        ));
        // Resume just after the NAME, not after the initializer: a binding declared
        // INSIDE another's initializer (`cva("…", { … })` bodies, IIFEs, arrow bodies)
        // is a binding too, and skipping the enclosing initializer would silently lose
        // every class string it holds.
        i = name_end;
    }
    found
}

/// Splits `cond ? then : else` at the top level (string- and bracket-aware,
/// skipping `?.` and `??`). Returns condition and both branches — the
/// condition is walked too, because in an argument list (`clsx('a', c ? 'b' :
/// 'c')`) everything before the `?` includes earlier class arguments.
fn split_ternary(expr: &str) -> Option<(&str, &str, &str)> {
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
                    return Some((&expr[..q], &expr[q + 1..i], &expr[i + 1..]));
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
    /// `@keyframes` blocks defined by the theme, keyed by animation name and
    /// stored as compact serialized CSS (`@keyframes pulse{50%{opacity:0.5}}`).
    keyframes: BTreeMap<String, String>,
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
            // `--<namespace>-*` is Tailwind's namespace wildcard, not a property
            // name: `--text-*: initial;` CLEARS every `--text-…` token declared so
            // far. That is how a theme replaces a whole scale instead of merging
            // into it — the mechanism a legacy v3 config needs, because its
            // resolved theme is complete and any leftover v4 default is wrong
            // (v4 gives `--text-5xl--line-height: 1`; a v3 config that sets
            // `fontSize: { '5xl': '2.5rem' }` gives the size NO line-height).
            let wildcard =
                j + 1 < bytes.len() && bytes[j] == b'*' && css[name_start..j].ends_with('-');
            let name_end = if wildcard { j + 1 } else { j };
            // Require a `:` right after the name (allowing whitespace).
            let mut k = name_end;
            while k < bytes.len() && bytes[k].is_ascii_whitespace() {
                k += 1;
            }
            if k >= bytes.len() || bytes[k] != b':' {
                i = j.max(name_start + 2);
                continue;
            }
            if wildcard {
                let Some(semi_rel) = css[k + 1..].find(';') else {
                    break;
                };
                // Only `initial` is meaningful (checked, and rejected, by
                // `validate_theme_wildcards` before any theme is parsed).
                let prefix = css[name_start..name_end - 1].to_string();
                values.retain(|name: &String, _: &mut String| !name.starts_with(&prefix));
                order.retain(|name: &String| !name.starts_with(&prefix));
                i = k + 1 + semi_rel + 1;
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
        let keyframes = parse_keyframes(css);
        Self {
            values,
            order,
            keyframes,
        }
    }

    fn contains(&self, name: &str) -> bool {
        self.values.contains_key(name)
    }

    /// Hard-errors on a `--<namespace>-*: <value>;` whose value is not `initial`.
    /// Tailwind gives the wildcard exactly one meaning (clear the namespace); any
    /// other value would otherwise be parsed as nothing at all and silently ship
    /// the scale the theme meant to replace.
    fn validate_wildcards(css: &str) -> Result<(), String> {
        let mut rest = css;
        while let Some(at) = rest.find("-*") {
            let after = rest[at + 2..].trim_start();
            let Some(value) = after.strip_prefix(':') else {
                rest = &rest[at + 2..];
                continue;
            };
            let end = value.find(';').unwrap_or(value.len());
            let value = value[..end].trim();
            if value != "initial" {
                let start = rest[..at].rfind("--").unwrap_or(0);
                return Err(format!(
                    "tailwind theme: `{}: {value};` — a `--…-*` namespace wildcard accepts \
                     only `initial` (it clears the namespace)",
                    rest[start..at + 2].trim()
                ));
            }
            rest = &rest[at + 2..];
        }
        Ok(())
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.values.get(name).map(|v| v.as_str())
    }

    /// The serialized `@keyframes` block for an animation name, if the theme
    /// defines one.
    fn keyframes(&self, name: &str) -> Option<&str> {
        self.keyframes.get(name).map(|v| v.as_str())
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

/// Extracts every `@keyframes <name> { … }` block from the theme CSS, keyed by
/// name, serialized compactly (whitespace collapsed around `{};:,`).
fn parse_keyframes(css: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let mut rest = css;
    while let Some(pos) = rest.find("@keyframes") {
        let after = &rest[pos + "@keyframes".len()..];
        let Some(open_rel) = after.find('{') else {
            break;
        };
        let name = after[..open_rel].trim().to_string();
        // Find the matching close brace for the block.
        let bytes = after.as_bytes();
        let mut depth = 0i32;
        let mut end = None;
        for (i, &b) in bytes.iter().enumerate().skip(open_rel) {
            match b {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(end) = end else { break };
        let body = &after[open_rel..=end];
        out.insert(
            name.clone(),
            format!("@keyframes {name}{}", compact_css(body)),
        );
        rest = &after[end + 1..];
    }
    out
}

/// Collapses a CSS block to a compact single-line form: whitespace runs become
/// one space, and spaces around `{`, `}`, `;`, `:` and `,` are dropped.
fn compact_css(block: &str) -> String {
    let mut out = String::with_capacity(block.len());
    let mut pending_space = false;
    for c in block
        .split_whitespace()
        .flat_map(|w| w.chars().chain(std::iter::once('\u{0}')))
    {
        if c == '\u{0}' {
            pending_space = true;
            continue;
        }
        if pending_space {
            let prev = out.chars().last();
            if !matches!(prev, None | Some('{' | '}' | ';' | ':' | ','))
                && !matches!(c, '{' | '}' | ';' | ':' | ',')
            {
                out.push(' ');
            }
            pending_space = false;
        }
        out.push(c);
    }
    // Drop `;` immediately before `}` (`opacity:0.5;}` -> `opacity:0.5}`).
    out.replace(";}", "}")
}

/// Collapses whitespace and rewrites `--theme(--x, initial)` (used by the default
/// font tokens) into `var(--x)`, matching the compiled theme layer. A pure
/// `calc(<number> / <number>)` whose quotient terminates (the line-height
/// ratios: `calc(1.5 / 1)` -> `1.5`) is constant-folded the way the reference
/// minifier folds it; non-terminating quotients keep the `calc()`.
fn normalize_theme_value(raw: &str) -> String {
    let collapsed = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    if let Some(rest) = collapsed.strip_prefix("--theme(") {
        let inner = rest.trim_end_matches(')');
        let first = inner.split(',').next().unwrap_or("").trim();
        return format!("var({first})");
    }
    if let Some(folded) = fold_exact_division(&collapsed) {
        return folded;
    }
    collapsed
}

/// Folds `calc(a / b)` over two plain decimal numbers exactly when the
/// reference minifier does. esbuild (Vite's CSS minifier) rewrites the
/// division as multiplication by the reciprocal and keeps the fold only when
/// that is lossless in f64 — `a * (1/b) == a / b` — which is why
/// `calc(1.5 / 1)` and `calc(2.25 / 1.875)` fold but `calc(1.75 / 1.25)`
/// stays (verified against esbuild directly).
fn fold_exact_division(value: &str) -> Option<String> {
    let inner = value.strip_prefix("calc(")?.strip_suffix(')')?;
    let (a, b) = inner.split_once('/')?;
    let a = a.trim();
    let b = b.trim();
    let is_number = |s: &str| !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit() || c == b'.');
    if !is_number(a) || !is_number(b) {
        return None;
    }
    let a: f64 = a.parse().ok()?;
    let b: f64 = b.parse().ok()?;
    if b == 0.0 || a * (1.0 / b) != a / b {
        return None;
    }
    let q = a / b;
    // Only fold when the quotient prints exactly at format_number's four
    // decimals.
    if (q * 10_000.0).round() / 10_000.0 != q {
        return None;
    }
    Some(format_number(q))
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
    RotateX,
    RotateY,
    RotateZ,
    SkewX,
    SkewY,
    SpaceXReverse,
    SpaceYReverse,
    ContainSize,
    ContainLayout,
    ContainPaint,
    ContainStyle,
    DivideXReverse,
    DivideYReverse,
    BorderSpacingX,
    BorderSpacingY,
    BorderStyle,
    GradientPosition,
    GradientFrom,
    GradientVia,
    GradientTo,
    GradientStops,
    GradientViaStops,
    GradientFromPosition,
    GradientViaPosition,
    GradientToPosition,
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
    Blur,
    Brightness,
    Contrast,
    Grayscale,
    HueRotate,
    Invert,
    FilterOpacity,
    Saturate,
    Sepia,
    DropShadow,
    DropShadowColor,
    DropShadowAlpha,
    DropShadowSize,
    BackdropBlur,
    BackdropBrightness,
    BackdropContrast,
    BackdropGrayscale,
    BackdropHueRotate,
    BackdropInvert,
    BackdropOpacity,
    BackdropSaturate,
    BackdropSepia,
    Duration,
    Ease,
    ScaleX,
    ScaleY,
    ScaleZ,
    OutlineStyle,
    Content,
    ScrollSnapStrictness,
    ScrollbarThumb,
    ScrollbarTrack,
    PanX,
    PanY,
    PinchZoom,
    TextShadowColor,
    TextShadowAlpha,
}

impl TwProp {
    /// `name, layer initial value, @property syntax, @property initial value`.
    fn spec(
        self,
    ) -> (
        &'static str,
        &'static str,
        &'static str,
        Option<&'static str>,
    ) {
        match self {
            TwProp::TranslateX => ("--tw-translate-x", "0", "\"*\"", Some("0")),
            TwProp::TranslateY => ("--tw-translate-y", "0", "\"*\"", Some("0")),
            TwProp::TranslateZ => ("--tw-translate-z", "0", "\"*\"", Some("0")),
            TwProp::RotateX => ("--tw-rotate-x", "initial", "\"*\"", None),
            TwProp::RotateY => ("--tw-rotate-y", "initial", "\"*\"", None),
            TwProp::RotateZ => ("--tw-rotate-z", "initial", "\"*\"", None),
            TwProp::SkewX => ("--tw-skew-x", "initial", "\"*\"", None),
            TwProp::SkewY => ("--tw-skew-y", "initial", "\"*\"", None),
            TwProp::SpaceXReverse => ("--tw-space-x-reverse", "0", "\"*\"", Some("0")),
            TwProp::SpaceYReverse => ("--tw-space-y-reverse", "0", "\"*\"", Some("0")),
            TwProp::ContainSize => ("--tw-contain-size", "initial", "\"*\"", None),
            TwProp::ContainLayout => ("--tw-contain-layout", "initial", "\"*\"", None),
            TwProp::ContainPaint => ("--tw-contain-paint", "initial", "\"*\"", None),
            TwProp::ContainStyle => ("--tw-contain-style", "initial", "\"*\"", None),
            TwProp::DivideXReverse => ("--tw-divide-x-reverse", "0", "\"*\"", Some("0")),
            TwProp::DivideYReverse => ("--tw-divide-y-reverse", "0", "\"*\"", Some("0")),
            TwProp::BorderSpacingX => ("--tw-border-spacing-x", "0", "\"<length>\"", Some("0")),
            TwProp::BorderSpacingY => ("--tw-border-spacing-y", "0", "\"<length>\"", Some("0")),
            TwProp::BorderStyle => ("--tw-border-style", "solid", "\"*\"", Some("solid")),
            TwProp::GradientPosition => ("--tw-gradient-position", "initial", "\"*\"", None),
            TwProp::GradientFrom => ("--tw-gradient-from", "#0000", "\"<color>\"", Some("#0000")),
            TwProp::GradientVia => ("--tw-gradient-via", "#0000", "\"<color>\"", Some("#0000")),
            TwProp::GradientTo => ("--tw-gradient-to", "#0000", "\"<color>\"", Some("#0000")),
            TwProp::GradientStops => ("--tw-gradient-stops", "initial", "\"*\"", None),
            TwProp::GradientViaStops => ("--tw-gradient-via-stops", "initial", "\"*\"", None),
            TwProp::GradientFromPosition => (
                "--tw-gradient-from-position",
                "0%",
                "\"<length-percentage>\"",
                Some("0%"),
            ),
            TwProp::GradientViaPosition => (
                "--tw-gradient-via-position",
                "50%",
                "\"<length-percentage>\"",
                Some("50%"),
            ),
            TwProp::GradientToPosition => (
                "--tw-gradient-to-position",
                "100%",
                "\"<length-percentage>\"",
                Some("100%"),
            ),
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
            TwProp::ShadowAlpha => (
                "--tw-shadow-alpha",
                "100%",
                "\"<percentage>\"",
                Some("100%"),
            ),
            TwProp::InsetShadow => ("--tw-inset-shadow", "0 0 #0000", "\"*\"", Some("0 0 #0000")),
            TwProp::InsetShadowColor => ("--tw-inset-shadow-color", "initial", "\"*\"", None),
            TwProp::InsetShadowAlpha => (
                "--tw-inset-shadow-alpha",
                "100%",
                "\"<percentage>\"",
                Some("100%"),
            ),
            TwProp::RingColor => ("--tw-ring-color", "initial", "\"*\"", None),
            TwProp::RingShadow => ("--tw-ring-shadow", "0 0 #0000", "\"*\"", Some("0 0 #0000")),
            TwProp::InsetRingColor => ("--tw-inset-ring-color", "initial", "\"*\"", None),
            TwProp::InsetRingShadow => (
                "--tw-inset-ring-shadow",
                "0 0 #0000",
                "\"*\"",
                Some("0 0 #0000"),
            ),
            TwProp::RingInset => ("--tw-ring-inset", "initial", "\"*\"", None),
            TwProp::RingOffsetWidth => ("--tw-ring-offset-width", "0px", "\"<length>\"", Some("0")),
            TwProp::RingOffsetColor => ("--tw-ring-offset-color", "#fff", "\"*\"", Some("#fff")),
            TwProp::RingOffsetShadow => (
                "--tw-ring-offset-shadow",
                "0 0 #0000",
                "\"*\"",
                Some("0 0 #0000"),
            ),
            TwProp::Blur => ("--tw-blur", "initial", "\"*\"", None),
            TwProp::Brightness => ("--tw-brightness", "initial", "\"*\"", None),
            TwProp::Contrast => ("--tw-contrast", "initial", "\"*\"", None),
            TwProp::Grayscale => ("--tw-grayscale", "initial", "\"*\"", None),
            TwProp::HueRotate => ("--tw-hue-rotate", "initial", "\"*\"", None),
            TwProp::Invert => ("--tw-invert", "initial", "\"*\"", None),
            TwProp::FilterOpacity => ("--tw-opacity", "initial", "\"*\"", None),
            TwProp::Saturate => ("--tw-saturate", "initial", "\"*\"", None),
            TwProp::Sepia => ("--tw-sepia", "initial", "\"*\"", None),
            TwProp::DropShadow => ("--tw-drop-shadow", "initial", "\"*\"", None),
            TwProp::DropShadowColor => ("--tw-drop-shadow-color", "initial", "\"*\"", None),
            TwProp::DropShadowAlpha => (
                "--tw-drop-shadow-alpha",
                "100%",
                "\"<percentage>\"",
                Some("100%"),
            ),
            TwProp::DropShadowSize => ("--tw-drop-shadow-size", "initial", "\"*\"", None),
            TwProp::BackdropBlur => ("--tw-backdrop-blur", "initial", "\"*\"", None),
            TwProp::BackdropBrightness => ("--tw-backdrop-brightness", "initial", "\"*\"", None),
            TwProp::BackdropContrast => ("--tw-backdrop-contrast", "initial", "\"*\"", None),
            TwProp::BackdropGrayscale => ("--tw-backdrop-grayscale", "initial", "\"*\"", None),
            TwProp::BackdropHueRotate => ("--tw-backdrop-hue-rotate", "initial", "\"*\"", None),
            TwProp::BackdropInvert => ("--tw-backdrop-invert", "initial", "\"*\"", None),
            TwProp::BackdropOpacity => ("--tw-backdrop-opacity", "initial", "\"*\"", None),
            TwProp::BackdropSaturate => ("--tw-backdrop-saturate", "initial", "\"*\"", None),
            TwProp::BackdropSepia => ("--tw-backdrop-sepia", "initial", "\"*\"", None),
            TwProp::Duration => ("--tw-duration", "initial", "\"*\"", None),
            TwProp::Ease => ("--tw-ease", "initial", "\"*\"", None),
            TwProp::ScaleX => ("--tw-scale-x", "1", "\"*\"", Some("1")),
            TwProp::ScaleY => ("--tw-scale-y", "1", "\"*\"", Some("1")),
            TwProp::ScaleZ => ("--tw-scale-z", "1", "\"*\"", Some("1")),
            TwProp::OutlineStyle => ("--tw-outline-style", "solid", "\"*\"", Some("solid")),
            TwProp::Content => ("--tw-content", "\"\"", "\"*\"", Some("\"\"")),
            TwProp::ScrollSnapStrictness => (
                "--tw-scroll-snap-strictness",
                "proximity",
                "\"*\"",
                Some("proximity"),
            ),
            TwProp::ScrollbarThumb => (
                "--tw-scrollbar-thumb",
                "#0000",
                "\"<color>\"",
                Some("#0000"),
            ),
            TwProp::ScrollbarTrack => (
                "--tw-scrollbar-track",
                "#0000",
                "\"<color>\"",
                Some("#0000"),
            ),
            TwProp::PanX => ("--tw-pan-x", "initial", "\"*\"", None),
            TwProp::PanY => ("--tw-pan-y", "initial", "\"*\"", None),
            TwProp::PinchZoom => ("--tw-pinch-zoom", "initial", "\"*\"", None),
            TwProp::TextShadowColor => ("--tw-text-shadow-color", "initial", "\"*\"", None),
            TwProp::TextShadowAlpha => (
                "--tw-text-shadow-alpha",
                "100%",
                "\"<percentage>\"",
                Some("100%"),
            ),
        }
    }

    /// Whether `dialect` declares this property with an EMPTY default rather than
    /// registering it with `@property`.
    ///
    /// v3's gradient stop positions default to nothing (`--tw-gradient-from-position: ;`)
    /// so that `from-cyan-500` composes to a bare `#06b6d4` with no position; v4
    /// registers them at `0%`/`50%`/`100%` and composes `#06b6d4 0%`. Compiling a
    /// v3 app the v4 way made `bg-gradient-to-r from-cyan-500 to-blue-500` compute
    /// `linear-gradient(to right, rgb(6,182,212) 0%, rgb(59,130,246) 100%)` where
    /// v3 computes `linear-gradient(to right, rgb(6,182,212), rgb(59,130,246))`.
    fn empty_default_in(self, dialect: Dialect) -> bool {
        dialect == Dialect::V3
            && matches!(
                self,
                TwProp::GradientFromPosition
                    | TwProp::GradientViaPosition
                    | TwProp::GradientToPosition
            )
    }

    fn layer_declaration(self) -> String {
        let (name, initial, _, _) = self.spec();
        format!("{name}:{initial}")
    }

    fn property_declaration(self) -> String {
        let (name, _, syntax, initial) = self.spec();
        match initial {
            Some(value) => {
                format!("@property {name}{{syntax:{syntax};inherits:false;initial-value:{value}}}")
            }
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

/// Registers the filter property group (Tailwind registers the whole group for
/// any `blur`/`brightness`/…/`drop-shadow` utility, because they compose into
/// one `filter`).
fn register_filter_group(tw_props: &mut BTreeSet<TwProp>) {
    for prop in [
        TwProp::Blur,
        TwProp::Brightness,
        TwProp::Contrast,
        TwProp::Grayscale,
        TwProp::HueRotate,
        TwProp::Invert,
        TwProp::FilterOpacity,
        TwProp::Saturate,
        TwProp::Sepia,
        TwProp::DropShadow,
        TwProp::DropShadowColor,
        TwProp::DropShadowAlpha,
        TwProp::DropShadowSize,
    ] {
        tw_props.insert(prop);
    }
}

/// Registers the backdrop-filter property group (any `backdrop-*` filter
/// utility registers all of them: they compose into one `backdrop-filter`).
fn register_backdrop_group(tw_props: &mut BTreeSet<TwProp>) {
    for prop in [
        TwProp::BackdropBlur,
        TwProp::BackdropBrightness,
        TwProp::BackdropContrast,
        TwProp::BackdropGrayscale,
        TwProp::BackdropHueRotate,
        TwProp::BackdropInvert,
        TwProp::BackdropOpacity,
        TwProp::BackdropSaturate,
        TwProp::BackdropSepia,
    ] {
        tw_props.insert(prop);
    }
}

/// Registers the gradient property group (any gradient position or `from-*`/
/// `via-*`/`to-*` stop registers all of them: they compose into one
/// `--tw-gradient-stops`).
///
/// v3 registers NO gradient property: it has no `--tw-gradient-position` or
/// `--tw-gradient-via-stops` at all, and its `--tw-gradient-from`/`-to` hold a
/// `<color> <position>` pair rather than a bare color — registering those with
/// v4's `syntax: "<color>"` would make every v3 stop declaration invalid. Its
/// only defaults are the three stop positions, declared empty (see
/// [`TwProp::empty_default_in`]).
fn register_gradient_group(tw_props: &mut BTreeSet<TwProp>, dialect: Dialect) {
    if dialect == Dialect::V3 {
        for prop in [
            TwProp::GradientFromPosition,
            TwProp::GradientViaPosition,
            TwProp::GradientToPosition,
        ] {
            tw_props.insert(prop);
        }
        return;
    }
    for prop in [
        TwProp::GradientPosition,
        TwProp::GradientFrom,
        TwProp::GradientVia,
        TwProp::GradientTo,
        TwProp::GradientStops,
        TwProp::GradientViaStops,
        TwProp::GradientFromPosition,
        TwProp::GradientViaPosition,
        TwProp::GradientToPosition,
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
    /// Additional pseudo suffixes, each producing another rule with the same
    /// declarations (e.g. `selection:` emits both `& ::selection` and
    /// `&::selection`; `marker:` emits four).
    extra_pseudos: Vec<String>,
    /// A selector fragment placed BEFORE the class selector, as a descendant
    /// ancestor (`in-<variant>` -> `:where(<fragment>) .class`).
    prefix: String,
    /// `:is(...)` clause inserted right after the class selector (group-hover).
    is_clause: String,
    /// A (prefix, suffix) pair wrapping the ENTIRE built selector, used by the
    /// child (`*:` -> `:is(& > *)`) and descendant (`**:` -> `:is(& *)`) variants.
    wrap: Option<(String, String)>,
    media: Vec<String>,
    order: u8,
    inject_content: bool,
}

/// Output order indices per variant, mirroring the reference stylesheet:
/// base < before/after < hover < focus < focus-visible < active < disabled
/// < breakpoints < dark.
/// Render a media-query condition the way Tailwind v4 formats it: a range query
/// `width>=40rem` -> `(width >= 40rem)` (the range operator MUST be space-separated to
/// be valid CSS), a feature query `hover:hover` -> `(hover: hover)`, and a bare feature
/// `print` -> `print` (no parentheses).
fn format_media_query(condition: &str) -> String {
    for op in ["<=", ">=", "<", ">"] {
        if let Some(pos) = condition.find(op) {
            let left = condition[..pos].trim();
            let right = condition[pos + op.len()..].trim();
            return format!("({left} {op} {right})");
        }
    }
    if let Some(pos) = condition.find(':') {
        let feature = condition[..pos].trim();
        let value = condition[pos + 1..].trim();
        return format!("({feature}: {value})");
    }
    condition.to_string()
}

/// The CSS pseudo-class for a Tailwind state name (used to compose `group-<state>` /
/// `peer-<state>` selectors). `None` for an unknown state.
fn state_pseudo(state: &str) -> Option<&'static str> {
    Some(match state {
        "hover" => ":hover",
        "focus" => ":focus",
        "focus-within" => ":focus-within",
        "focus-visible" => ":focus-visible",
        "active" => ":active",
        "visited" => ":visited",
        "target" => ":target",
        "disabled" => ":disabled",
        "enabled" => ":enabled",
        "checked" => ":checked",
        "indeterminate" => ":indeterminate",
        "default" => ":default",
        "required" => ":required",
        "optional" => ":optional",
        "valid" => ":valid",
        "invalid" => ":invalid",
        "in-range" => ":in-range",
        "out-of-range" => ":out-of-range",
        "read-only" => ":read-only",
        "read-write" => ":read-write",
        "placeholder-shown" => ":placeholder-shown",
        "autofill" => ":autofill",
        "empty" => ":empty",
        "user-valid" => ":user-valid",
        "user-invalid" => ":user-invalid",
        "inert" => ":is([inert], [inert] *)",
        "open" => ":is([open], :popover-open, :open)",
        "first" => ":first-child",
        "last" => ":last-child",
        "only" => ":only-child",
        "odd" => ":nth-child(odd)",
        "even" => ":nth-child(even)",
        "first-of-type" => ":first-of-type",
        "last-of-type" => ":last-of-type",
        "only-of-type" => ":only-of-type",
        _ => return None,
    })
}

/// The `:is(:where(.group[/name])<state-pseudo><combinator>)` clause for a
/// `group-<state>` / `peer-<state>` variant (with an optional `/name` selecting a
/// named marker class). `peer` uses the following-sibling combinator `~ *`; a
/// `group` uses the descendant combinator ` *`. `None` for an unknown state.
fn group_peer_clause(v: &str) -> Option<String> {
    let (marker, rest) = v.split_once('-')?;
    let (state, name) = match rest.split_once('/') {
        Some((s, n)) => (s, Some(n)),
        None => (rest, None),
    };
    let pseudo = state_pseudo(state)?;
    let marker_class = match name {
        Some(n) => format!("{marker}/{n}"),
        None => marker.to_string(),
    };
    let escaped = escape_class(&marker_class);
    let combinator = if marker == "peer" { " ~ *" } else { " *" };
    Some(format!(":is(:where(.{escaped}){pseudo}{combinator})"))
}

/// The selector fragment a variant contributes when composed inside `not-<v>`,
/// `has-<v>`, or `in-<v>`. `checked` -> `:checked`; `group-focus/name` ->
/// `:is(:where(.group\/name):focus *)`; nesting like `has-checked` composes.
/// `None` when the inner variant has no such composable selector form.
fn variant_fragment(v: &str) -> Option<String> {
    if let Some(pseudo) = state_pseudo(v) {
        return Some(pseudo.to_string());
    }
    if v.starts_with("group-") || v.starts_with("peer-") {
        return group_peer_clause(v);
    }
    if let Some(inner) = v.strip_prefix("has-") {
        return Some(format!(":has({})", variant_fragment(inner)?));
    }
    if let Some(inner) = v.strip_prefix("not-") {
        return Some(format!(":not({})", variant_fragment(inner)?));
    }
    None
}

/// The value inside an `nth-*` positional pseudo: a bare positive integer
/// (`3`) or an arbitrary `[…]` microsyntax value (`[2n+1]`). `None` (rejected)
/// for negatives, non-numeric bare words, or `/`-modified tokens.
fn nth_value(s: &str) -> Option<String> {
    if let Some(inner) = s.strip_prefix('[').and_then(|x| x.strip_suffix(']')) {
        return Some(inner.to_string());
    }
    if !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) {
        return Some(s.to_string());
    }
    None
}

/// The `@container [name] (width OP value)` at-rule string for a container-query
/// variant (the leading `@` already stripped). Accepts `@lg`/`@sm`/… (theme
/// container sizes, `>=`), `@min-<size>`/`@max-<size>`, `@min-[…]`/`@max-[…]`,
/// and a trailing `/name` selecting a named container. `None` (rejected) when the
/// size token is not a known container token nor an arbitrary `[…]` value.
fn container_query_condition(variant: &str, theme: &Theme) -> Option<String> {
    let (core, name) = match variant.split_once('/') {
        Some((c, n)) => (c, Some(n)),
        None => (variant, None),
    };
    let (op, size_token) = if let Some(rest) = core.strip_prefix("min-") {
        (">=", rest)
    } else if let Some(rest) = core.strip_prefix("max-") {
        ("<", rest)
    } else {
        (">=", core)
    };
    if size_token.is_empty() {
        return None;
    }
    let value = if let Some(inner) = size_token
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
    {
        inner.to_string()
    } else {
        theme.get(&format!("--container-{size_token}"))?.to_string()
    };
    let name_part = match name {
        Some(n) if !n.is_empty() => format!("{n} "),
        Some(_) => return None,
        None => String::new(),
    };
    Some(format!("@container {name_part}(width {op} {value})"))
}

fn parse_variants(
    segments: &[&str],
    class: &str,
    theme: &Theme,
    tw_props: &mut BTreeSet<TwProp>,
    custom_variants: &std::collections::BTreeMap<String, String>,
) -> Result<VariantSpec, Fail> {
    let mut spec = VariantSpec {
        pseudo: String::new(),
        extra_pseudos: Vec::new(),
        prefix: String::new(),
        is_clause: String::new(),
        wrap: None,
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
            // Structural / form / state pseudo-CLASSES: append the pseudo verbatim.
            // A few Tailwind names diverge from the CSS pseudo (`first` ->
            // `:first-child`), so map those explicitly.
            "checked" | "enabled" | "default" | "indeterminate" | "required" | "valid"
            | "invalid" | "in-range" | "out-of-range" | "read-only" | "read-write" | "autofill"
            | "placeholder-shown" | "focus-within" | "visited" | "target" | "empty"
            | "optional" | "user-valid" | "user-invalid" | "popover-open" | "first-of-type"
            | "last-of-type" | "only-of-type" => {
                spec.pseudo.push(':');
                spec.pseudo.push_str(variant);
                spec.order = spec.order.max(5);
            }
            "first" => {
                spec.pseudo.push_str(":first-child");
                spec.order = spec.order.max(5);
            }
            "last" => {
                spec.pseudo.push_str(":last-child");
                spec.order = spec.order.max(5);
            }
            "only" => {
                spec.pseudo.push_str(":only-child");
                spec.order = spec.order.max(5);
            }
            "odd" => {
                spec.pseudo.push_str(":nth-child(odd)");
                spec.order = spec.order.max(5);
            }
            "even" => {
                spec.pseudo.push_str(":nth-child(even)");
                spec.order = spec.order.max(5);
            }
            "open" => {
                spec.pseudo.push_str(":is([open], :popover-open, :open)");
                spec.order = spec.order.max(5);
            }
            "inert" => {
                spec.pseudo.push_str(":is([inert], [inert] *)");
                spec.order = spec.order.max(5);
            }
            // Child (`*:`) and descendant (`**:`) variants: wrap the whole
            // candidate selector in `:is(& > *)` / `:is(& *)`.
            "*" => {
                spec.wrap = Some((":is(".to_string(), " > *)".to_string()));
                spec.order = spec.order.max(5);
            }
            "**" => {
                spec.wrap = Some((":is(".to_string(), " *)".to_string()));
                spec.order = spec.order.max(5);
            }
            // Pseudo-ELEMENTS.
            "first-letter" | "first-line" | "placeholder" | "details-content" => {
                spec.pseudo.push_str("::");
                spec.pseudo.push_str(variant);
                spec.order = spec.order.max(1);
            }
            "file" => {
                spec.pseudo.push_str("::file-selector-button");
                spec.order = spec.order.max(1);
            }
            // Media-feature variants: no selector change, a media wrapper (formatted by
            // `format_media_query`). `print` renders parenthesis-free.
            "motion-safe" => {
                push_media(&mut spec.media, "prefers-reduced-motion:no-preference");
                spec.order = spec.order.max(6);
            }
            "motion-reduce" => {
                push_media(&mut spec.media, "prefers-reduced-motion:reduce");
                spec.order = spec.order.max(6);
            }
            "contrast-more" => {
                push_media(&mut spec.media, "prefers-contrast:more");
                spec.order = spec.order.max(6);
            }
            "contrast-less" => {
                push_media(&mut spec.media, "prefers-contrast:less");
                spec.order = spec.order.max(6);
            }
            "inverted-colors" => {
                push_media(&mut spec.media, "inverted-colors:inverted");
                spec.order = spec.order.max(6);
            }
            "forced-colors" => {
                push_media(&mut spec.media, "forced-colors:active");
                spec.order = spec.order.max(6);
            }
            "portrait" => {
                push_media(&mut spec.media, "orientation:portrait");
                spec.order = spec.order.max(6);
            }
            "landscape" => {
                push_media(&mut spec.media, "orientation:landscape");
                spec.order = spec.order.max(6);
            }
            "print" => {
                push_media(&mut spec.media, "print");
                spec.order = spec.order.max(6);
            }
            // Direction variants: match the element's resolved direction, both via
            // `:dir()` and the legacy `[dir=…]` attribute (self and descendant).
            "ltr" => {
                spec.pseudo
                    .push_str(":where(:dir(ltr), [dir=\"ltr\"], [dir=\"ltr\"] *)");
                spec.order = spec.order.max(5);
            }
            "rtl" => {
                spec.pseudo
                    .push_str(":where(:dir(rtl), [dir=\"rtl\"], [dir=\"rtl\"] *)");
                spec.order = spec.order.max(5);
            }
            // `noscript:` — the `(scripting: none)` media feature.
            "noscript" => {
                push_media(&mut spec.media, "scripting:none");
                spec.order = spec.order.max(6);
            }
            // any-pointer media-feature variants (`any-pointer-coarse/fine/none`).
            "any-pointer-coarse" | "any-pointer-fine" | "any-pointer-none" => {
                push_media(&mut spec.media, &format!("any-pointer:{}", &variant[12..]));
                spec.order = spec.order.max(6);
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
            // `selection:` emits the utility under both `& ::selection` (any
            // descendant) and `&::selection` (the element itself).
            "selection" => {
                spec.pseudo.push_str(" ::selection");
                spec.extra_pseudos = vec!["::selection".to_string()];
                spec.order = spec.order.max(1);
            }
            // `marker:` emits the utility under the `::marker` and legacy
            // `::-webkit-details-marker` pseudo-elements, each both as a
            // descendant (` ::marker`) and directly (`::marker`): four rules.
            "marker" => {
                spec.pseudo.push_str(" ::-webkit-details-marker");
                spec.extra_pseudos = vec![
                    " ::marker".to_string(),
                    "::-webkit-details-marker".to_string(),
                    "::marker".to_string(),
                ];
                spec.order = spec.order.max(1);
            }
            // `@starting-style` transition variant: a bare at-rule wrapper (no
            // condition), carried through the media list with an `@` sentinel.
            "starting" => {
                push_media(&mut spec.media, "@starting-style");
                spec.order = spec.order.max(1);
            }
            // These built-in variants take no `/modifier`; Tailwind rejects
            // `backdrop/foo`, `contrast-more/foo`, `contrast-less/foo` and
            // generates nothing.
            v if matches!(
                v.split_once('/').map(|(name, _)| name),
                Some("backdrop")
                    | Some("contrast-more")
                    | Some("contrast-less")
                    | Some("selection")
                    | Some("marker")
                    | Some("placeholder")
                    | Some("placeholder-shown")
                    | Some("inert")
                    | Some("ltr")
                    | Some("rtl")
                    | Some("noscript")
            ) =>
            {
                return Err(Fail::Invalid);
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
                    Fail::Unsupported(format!("unknown breakpoint `{bp}:` in class `{class}`"))
                })?;
                push_media(&mut spec.media, &format!("width>={value}"));
                let index = ["sm", "md", "lg", "xl", "2xl"]
                    .iter()
                    .position(|b| *b == bp)
                    .unwrap() as u8;
                spec.order = spec.order.max(7 + index);
            }
            // not-<variant>: negate the inner variant's selector fragment,
            // appended as a `:not(...)` pseudo. `not-open` -> `:not(:is([open], …))`.
            // Tailwind rejects it when the inner variant has no composable form.
            v if v.starts_with("not-") => {
                let frag = variant_fragment(&v[4..]).ok_or(Fail::Invalid)?;
                spec.pseudo.push_str(&format!(":not({frag})"));
                spec.order = spec.order.max(5);
            }
            // has-<variant>: `:has(<inner-fragment>)`. `has-checked` -> `:has(:checked)`.
            v if v.starts_with("has-") => {
                let frag = variant_fragment(&v[4..]).ok_or(Fail::Invalid)?;
                spec.pseudo.push_str(&format!(":has({frag})"));
                spec.order = spec.order.max(5);
            }
            // in-<variant>: the inner fragment becomes a `:where(...)` ancestor
            // placed before the class selector.
            v if v.starts_with("in-") => {
                let frag = variant_fragment(&v[3..]).ok_or(Fail::Invalid)?;
                spec.prefix = format!(":where({frag}) ");
                spec.order = spec.order.max(5);
            }
            // nth-<n> / nth-last-<n> / nth-of-type-<n> / nth-last-of-type-<n>:
            // a positional pseudo taking a bare positive integer or `[…]` value.
            v if v.starts_with("nth-") => {
                let rest = &v[4..];
                let (func, num) = if let Some(n) = rest.strip_prefix("last-of-type-") {
                    (":nth-last-of-type", n)
                } else if let Some(n) = rest.strip_prefix("of-type-") {
                    (":nth-of-type", n)
                } else if let Some(n) = rest.strip_prefix("last-") {
                    (":nth-last-child", n)
                } else {
                    (":nth-child", rest)
                };
                let value = nth_value(num).ok_or(Fail::Invalid)?;
                spec.pseudo.push_str(&format!("{func}({value})"));
                spec.order = spec.order.max(5);
            }
            // supports-<feature> / supports-[…]: an `@supports` feature query. A
            // bare feature name probes `(<feature>: var(--tw))`; an arbitrary
            // value is used verbatim (a full declaration or bare property name).
            v if v.starts_with("supports-") => {
                let rest = &v[9..];
                if rest.contains('/') {
                    return Err(Fail::Invalid);
                }
                let cond =
                    if let Some(inner) = rest.strip_prefix('[').and_then(|x| x.strip_suffix(']')) {
                        if inner.contains(':') {
                            inner.to_string()
                        } else {
                            format!("{inner}: var(--tw)")
                        }
                    } else if rest.is_empty() {
                        return Err(Fail::Invalid);
                    } else {
                        format!("{rest}: var(--tw)")
                    };
                push_media(&mut spec.media, &format!("@supports ({cond})"));
                spec.order = spec.order.max(6);
            }
            // group-<state> / peer-<state>: the state pseudo inside a `:is()` clause
            // rooted at a `.group`/`.peer` ancestor (peer is a following sibling `~`).
            v if v.starts_with("group-") || v.starts_with("peer-") => {
                spec.is_clause = group_peer_clause(v).ok_or_else(|| {
                    Fail::Unsupported(format!(
                        "unsupported `{v}:` variant in `{class}`: unknown group/peer state"
                    ))
                })?;
                let state = v.split_once('-').unwrap().1;
                if state.split('/').next() == Some("hover") {
                    push_media(&mut spec.media, "hover:hover");
                }
                spec.order = spec.order.max(2);
            }
            // min-<bp> / max-<bp>: width range media queries from the theme breakpoints.
            v if v.starts_with("min-") || v.starts_with("max-") => {
                let (kind, bp) = v.split_once('-').unwrap();
                let value = theme.get(&format!("--breakpoint-{bp}")).ok_or_else(|| {
                    Fail::Unsupported(format!("unknown breakpoint `{v}:` in `{class}`"))
                })?;
                let op = if kind == "min" { ">=" } else { "<" };
                push_media(&mut spec.media, &format!("width{op}{value}"));
                spec.order = spec.order.max(6);
            }
            // aria-<state>: `[aria-<state>="true"]` for the boolean ARIA states.
            // The variant takes no `/modifier`; Tailwind rejects `aria-checked/foo`
            // (generates nothing).
            v if v.starts_with("aria-") && !v[5..].starts_with('[') => {
                if v[5..].contains('/') {
                    return Err(Fail::Invalid);
                }
                spec.pseudo
                    .push_str(&format!("[aria-{}=\"true\"]", &v[5..]));
                spec.order = spec.order.max(5);
            }
            // data-<name>: `[data-<name>]` attribute presence.
            v if v.starts_with("data-") && !v[5..].starts_with('[') => {
                spec.pseudo.push_str(&format!("[data-{}]", &v[5..]));
                spec.order = spec.order.max(5);
            }
            // pointer / any-pointer media-feature variants.
            v if v.starts_with("pointer-") => {
                push_media(&mut spec.media, &format!("pointer:{}", &v[8..]));
                spec.order = spec.order.max(6);
            }
            // Container-query variants: `@sm`/`@lg`/…, `@min-[…]`/`@max-[…]`,
            // `@min-lg`/`@max-lg`, and named `@lg/name`. Each wraps the rule in an
            // `@container [name] (width >= … | width < …)` at-rule.
            v if v.starts_with('@') => {
                let cond = container_query_condition(&v[1..], theme).ok_or(Fail::Invalid)?;
                push_media(&mut spec.media, &cond);
                spec.order = spec.order.max(6);
            }
            other => {
                // `!` can never appear in a variant segment (the important
                // marker is only legal on the utility itself), so a candidate
                // like the malformed `hover:!dark:bg-rose-400` is one Tailwind
                // rejects outright and generates nothing for. Anything shaped
                // like a real variant stays a hard engine-gap error.
                if other.is_empty() || other.contains('!') {
                    return Err(Fail::Invalid);
                }
                return Err(Fail::Unsupported(format!(
                    "unsupported Tailwind variant `{other}:` in class `{class}`: the native compiler does not yet generate it. Extend src/tailwind.rs (do not silently drop it)."
                )));
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
fn utility_root_recognized(class: &str, custom_utilities: &CustomUtilities) -> bool {
    let segments = split_variants(class);
    let base = segments.last().copied().unwrap_or(class);
    let base = base.strip_prefix('!').unwrap_or(base);
    let base = base.strip_suffix('!').unwrap_or(base);
    let base = base.strip_prefix('-').unwrap_or(base);
    // A candidate the app's own `@utility` claims is a utility BY DEFINITION —
    // an engine gap in its body must never be filtered out as a stray token.
    if custom_utilities.lookup(base).is_some() {
        return true;
    }
    const PREFIXES: &[&str] = &[
        "bg-",
        "text-",
        "font-",
        "border-",
        "rounded-",
        "ring-",
        "outline-",
        "shadow-",
        "inset-",
        "top-",
        "right-",
        "bottom-",
        "left-",
        "z-",
        "translate-",
        "rotate-",
        "scale-",
        "skew-",
        "w-",
        "h-",
        "min-w-",
        "min-h-",
        "max-w-",
        "max-h-",
        "size-",
        "m-",
        "mx-",
        "my-",
        "mt-",
        "mr-",
        "mb-",
        "ml-",
        "p-",
        "px-",
        "py-",
        "pt-",
        "pr-",
        "pb-",
        "pl-",
        "gap-",
        "space-",
        "flex-",
        "grid-",
        "col-",
        "row-",
        "justify-",
        "items-",
        "content-",
        "self-",
        "place-",
        "order-",
        "leading-",
        "tracking-",
        "align-",
        "list-",
        "decoration-",
        "underline-",
        "overflow-",
        "overscroll-",
        "cursor-",
        "select-",
        "pointer-events-",
        "opacity-",
        "transition-",
        "duration-",
        "delay-",
        "ease-",
        "animate-",
        "fill-",
        "stroke-",
        "object-",
        "aspect-",
        "columns-",
        "break-",
        "whitespace-",
        "line-clamp-",
        "backdrop-",
        "blur-",
        "brightness-",
        "contrast-",
        "divide-",
        "accent-",
        "caret-",
        "scroll-",
        "snap-",
        "touch-",
        "will-change-",
        "from-",
        "via-",
        "to-",
        "drop-shadow-",
        "transform-",
        "origin-",
        "perspective-",
        "clear-",
        "float-",
        "scheme-",
        "contain-",
        "backface-",
        "forced-color-adjust-",
    ];
    const EXACT: &[&str] = &[
        "flex",
        "grid",
        "block",
        "inline",
        "inline-block",
        "inline-flex",
        "inline-grid",
        "hidden",
        "contents",
        "static",
        "fixed",
        "absolute",
        "relative",
        "sticky",
        "isolate",
        "visible",
        "invisible",
        "container",
        "italic",
        "not-italic",
        "underline",
        "overline",
        "line-through",
        "no-underline",
        "uppercase",
        "lowercase",
        "capitalize",
        "normal-case",
        "truncate",
        "antialiased",
        "subpixel-antialiased",
        "rounded",
        "rounded-full",
        "border",
        "ring",
        "ring-inset",
        "shadow",
        "outline",
        "outline-none",
        "transition",
        "grow",
        "shrink",
        "tabular-nums",
        "sr-only",
        "not-sr-only",
        "group",
        "peer",
        "table",
        "inline-table",
        "table-caption",
        "table-cell",
        "table-column",
        "table-column-group",
        "table-footer-group",
        "table-header-group",
        "table-row",
        "table-row-group",
        "flow-root",
        "box-border",
        "box-content",
        "transform",
        "drop-shadow",
        "backdrop-blur",
    ];
    PREFIXES.iter().any(|prefix| base.starts_with(prefix)) || EXACT.contains(&base)
}

/// Renders a full utility class (with any variant chain) into a CSS rule string
/// plus its output grouping. Errors if the token is not a recognized utility.
/// Splits Tailwind's important marker off a utility name: `bg-black!` (the v4
/// spelling) or the legacy `!bg-black` (v3). The marker is part of the CANDIDATE,
/// not of the utility, so every place that resolves a utility name — a scanned
/// class, an `@apply` in a rule, an `@apply` inside an `@utility` body — has to
/// strip it before the lookup and add `!important` to what comes back.
fn split_important(base: &str) -> (&str, bool) {
    if let Some(rest) = base.strip_prefix('!') {
        (rest, true)
    } else if let Some(rest) = base.strip_suffix('!') {
        (rest, true)
    } else {
        (base, false)
    }
}

/// Appends `!important` to a declaration value, unless the value already
/// carries it (a `@utility` body may write it literally).
fn with_important(value: &str, important: bool) -> String {
    if important && !value.contains("!important") {
        format!("{value}!important")
    } else {
        value.to_string()
    }
}

fn render_utility(
    class: &str,
    theme: &Theme,
    tw_props: &mut BTreeSet<TwProp>,
    custom_variants: &std::collections::BTreeMap<String, String>,
    custom_utilities: &CustomUtilities,
    dialect: Dialect,
) -> Result<RenderedRule, Fail> {
    // Split on `:` only outside brackets: arbitrary values may contain `:`
    // (e.g. `bg-[color:var(--x)]`).
    let mut segments = split_variants(class);
    let base = segments
        .pop()
        .ok_or_else(|| Fail::Unsupported(format!("empty class candidate `{class}`")))?;
    let spec = parse_variants(&segments, class, theme, tw_props, custom_variants)?;

    // The important marker: `!` prefixing the utility (v4) or, legacy, at its
    // end. Appends `!important` to every declaration.
    let (base, important) = split_important(base);

    // An app-defined `@utility` wins over the built-in of the same name — that
    // is what registering one means. Its body may nest, so it is emitted as raw
    // nested CSS (exactly the shape Tailwind emits) rather than flat declarations.
    if !custom_utilities.is_empty()
        && let Some(found) = custom_utilities.lookup(base)
    {
        let bang = if important { "!important" } else { "" };
        let expanded = expand_utility_body(
            found.body,
            base,
            found.value,
            found.modifier,
            theme,
            tw_props,
            dialect,
            bang,
        )?;
        // Every declaration dropped: this candidate's value satisfies none of the
        // utility's `--value(…)` forms, so Tailwind generates nothing for it.
        if expanded.is_empty() {
            return Err(Fail::Invalid);
        }
        let escaped = escape_class(class);
        let mut body = String::new();
        for (prop, value) in &expanded.decls {
            body.push_str(&format!("{prop}:{value}{bang};"));
        }
        body.push_str(&expanded.nested);
        let mut css = String::new();
        for pseudo in std::iter::once(&spec.pseudo).chain(spec.extra_pseudos.iter()) {
            let core = format!("{}.{escaped}{}{}", spec.prefix, spec.is_clause, pseudo);
            let selector = match &spec.wrap {
                Some((pre, post)) => format!("{pre}{core}{post}"),
                None => core,
            };
            css.push_str(&format!("{selector}{{{body}}}"));
        }
        return Ok(RenderedRule {
            css,
            order: spec.order,
            media_key: spec.media.join("|"),
            rank: 100,
        });
    }

    // The `container` utility is a single rule (`width:100%`) carrying a nested
    // `@media (width >= <bp>) { max-width: <bp> }` block for every theme
    // breakpoint, in the theme's source order. It is emitted as raw nested CSS
    // rather than through the flat declaration path.
    if base == "container" {
        let escaped = escape_class(class);
        let core = format!("{}.{escaped}{}{}", spec.prefix, spec.is_clause, spec.pseudo);
        let selector = match &spec.wrap {
            Some((pre, post)) => format!("{pre}{core}{post}"),
            None => core,
        };
        let bang = if important { "!important" } else { "" };
        let mut body = format!("width:100%{bang};");
        for name in &theme.order {
            if name.starts_with("--breakpoint-")
                && let Some(value) = theme.get(name)
            {
                body.push_str(&format!(
                    "@media (width >= {value}){{max-width:{value}{bang}}}"
                ));
            }
        }
        return Ok(RenderedRule {
            css: format!("{selector}{{{body}}}"),
            order: spec.order,
            media_key: spec.media.join("|"),
            rank: 100,
        });
    }

    let utility = generate_utility(base, class, theme, tw_props, dialect)?;
    let escaped = escape_class(class);
    let build_selector = |pseudo: &str| {
        let core = match &utility.selector {
            SelectorKind::Class => {
                format!("{}.{escaped}{}{}", spec.prefix, spec.is_clause, pseudo)
            }
            SelectorKind::ClassPseudoElement(pe) => {
                format!(
                    "{}.{escaped}{}{}{}",
                    spec.prefix, spec.is_clause, pseudo, pe
                )
            }
            // v3 targets every child but the FIRST (`> :not([hidden]) ~ :not([hidden])`)
            // and v4 every child but the LAST — which is why their `space-*`/`divide-*`
            // margins/widths sit on opposite edges (see the `space_*` decls below).
            SelectorKind::SpaceChildren if dialect == Dialect::V3 => {
                format!(".{escaped} > :not([hidden]) ~ :not([hidden])")
            }
            SelectorKind::SpaceChildren => {
                format!(":where(.{escaped} > :not(:last-child))")
            }
        };
        match &spec.wrap {
            Some((pre, post)) => format!("{pre}{core}{post}"),
            None => core,
        }
    };
    let mut decls = utility.decls;
    if spec.inject_content && !decls.iter().any(|(prop, _)| prop == "content") {
        decls.insert(0, ("content".to_string(), "var(--tw-content)".to_string()));
    }
    let bang = if important { "!important" } else { "" };
    let body = decls
        .iter()
        .map(|(prop, value)| format!("{prop}:{value}{bang}"))
        .collect::<Vec<_>>()
        .join(";");
    // Most variants produce one rule; a few (e.g. `selection:`) emit the same
    // declarations under a second selector.
    let mut css = format!("{}{{{}}}", build_selector(&spec.pseudo), body);
    for extra in &spec.extra_pseudos {
        css.push_str(&format!("{}{{{}}}", build_selector(extra), body));
    }
    Ok(RenderedRule {
        css,
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
    /// `.class<pseudo-element>` — a utility that bakes a pseudo-element into its
    /// own selector (e.g. `placeholder-<color>` targets `::placeholder`).
    ClassPseudoElement(String),
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
/// The container-type/container-name utilities: `@container` (inline-size),
/// `@container-normal`, `@container-size`, each optionally carrying a `/name`
/// modifier that also sets `container-name`. Emits `container-type` first, then
/// `container-name` (matching the reference output). `None` when `base` is not
/// one of these; an out-of-shape `@container-…` returns `Some(Err(Invalid))`.
fn container_type_utility(base: &str) -> Option<Result<Utility, Fail>> {
    let rest = base.strip_prefix('@')?;
    let (core, name) = match rest.split_once('/') {
        Some((c, n)) => (c, Some(n)),
        None => (rest, None),
    };
    let container_type = match core {
        "container" => "inline-size",
        "container-normal" => "normal",
        "container-size" => "size",
        _ => return None,
    };
    if matches!(name, Some("")) {
        return Some(Err(Fail::Invalid));
    }
    let mut decls: Vec<(&'static str, String)> =
        vec![("container-type", container_type.to_string())];
    if let Some(n) = name {
        decls.push(("container-name", n.to_string()));
    }
    Some(Ok(Utility::simple(decls)))
}

fn generate_utility(
    base: &str,
    full: &str,
    theme: &Theme,
    tw_props: &mut BTreeSet<TwProp>,
    dialect: Dialect,
) -> Result<Utility, Fail> {
    // A candidate ending in `-` is a template-literal fragment
    // (`grid-cols-${n}` scans as `grid-cols-`); Tailwind's candidate parser
    // rejects it and generates nothing.
    if base.ends_with('-') || base.is_empty() {
        return Err(Fail::Invalid);
    }

    // container-type / container-name: `@container`, `@container-normal`,
    // `@container-size`, each with an optional `/name` naming the container.
    if let Some(result) = container_type_utility(base) {
        return result;
    }

    // Scroll / overflow / snap / blend / break / box / columns / object family.
    if let Some(result) = scroll_overflow_utility(base, theme, tw_props) {
        return result;
    }

    // Static keyword utilities. None of them accept a `/<modifier>`; Tailwind
    // rejects `flex/foo`, `select-none/foo`, `text-wrap/foo`, … (generates
    // nothing).
    {
        let (kw_core, kw_has_mod) = match base.split_once('/') {
            Some((core, _)) => (core, true),
            None => (base, false),
        };
        if let Some(decls) = keyword_utility(kw_core) {
            if kw_has_mod {
                return Err(Fail::Invalid);
            }
            return Ok(Utility::simple(decls));
        }
    }

    // Alignment: justify-content/items/self, align-content/items/self,
    // place-content/items/self. Returns `None` (falls through) when `base` is not
    // one of these — notably `content-[…]` defers to the generic content handler.
    if let Some(result) = alignment_utility(base) {
        return result;
    }

    // tab-size: `tab-<integer>` (bare positive integers) and `tab-[<value>]`.
    // Non-integer / fraction / `/modifier` forms are rejected (generate nothing).
    if let Some(v) = base.strip_prefix("tab-") {
        if is_bare_integer(v) {
            return Ok(Utility::simple(vec![("tab-size", v.to_string())]));
        }
        if let Some(inner) = arbitrary_value(v) {
            return Ok(Utility::simple(vec![("tab-size", inner)]));
        }
        return Err(Fail::Invalid);
    }

    // zoom: `zoom-<integer>` → `zoom:<n>%` and `zoom-[<value>]`. Non-integer /
    // negative / `/modifier` forms are rejected (generate nothing).
    if let Some(v) = base.strip_prefix("zoom-") {
        if is_bare_integer(v) {
            return Ok(Utility::simple(vec![("zoom", format!("{v}%"))]));
        }
        if let Some(inner) = arbitrary_value(v) {
            return Ok(Utility::simple(vec![("zoom", inner)]));
        }
        return Err(Fail::Invalid);
    }

    // A leading `-` on a non-negatable static keyword utility (e.g. `-contents`,
    // `-flex`, `-block`) is rejected by Tailwind and generates nothing.
    if let Some(rest) = base.strip_prefix('-')
        && keyword_utility(rest).is_some()
    {
        return Err(Fail::Invalid);
    }

    // not-sr-only: the inverse of sr-only, restoring normal flow.
    if base == "not-sr-only" {
        return Ok(Utility::simple(vec![
            ("position", "static".to_string()),
            ("width", "auto".to_string()),
            ("height", "auto".to_string()),
            ("padding", "0".to_string()),
            ("margin", "0".to_string()),
            ("overflow", "visible".to_string()),
            ("clip-path", "none".to_string()),
            ("white-space", "normal".to_string()),
        ]));
    }
    // A leading `-` on the sr-only composites is rejected (generates nothing).
    if base == "-not-sr-only" || base == "-sr-only" {
        return Err(Fail::Invalid);
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

    // The `mask-*` family (gradient masks, edge masks, clip/origin/position/type).
    // Handled before the generic negative gate below because `mask-linear-*` and
    // `mask-conic-*` accept a leading `-` for a negative angle.
    if positive_base == "mask" || positive_base.starts_with("mask-") {
        return mask_utility(positive_base, negative);
    }

    // The transform family: translate / scale / rotate / skew / transform /
    // perspective / (perspective-)origin. All share the leading `-` negative
    // strip above; each rejects a leading `-` where Tailwind does.
    if let Some(result) = transform_family_utility(positive_base, negative, theme, tw_props) {
        return result;
    }

    // inset-shadow / inset-shadow-<size> / inset-shadow-none / inset-shadow-<color>:
    // the inset box-shadow slot. No negative form (a leading `-` falls through to
    // the position-offset family, which rejects it). Bare `inset-shadow` is not a
    // utility. Handled before the position-offset family so `inset-shadow-*` is not
    // consumed as an `inset` offset value.
    if !negative && positive_base == "inset-shadow" {
        return Err(Fail::Invalid);
    }
    if !negative && let Some(size) = positive_base.strip_prefix("inset-shadow-") {
        if size == "none" {
            register_shadow_group(tw_props);
            return Ok(Utility::simple(vec![
                ("--tw-inset-shadow", "inset 0 0 #0000".to_string()),
                ("box-shadow", dialect.box_shadow_chain().to_string()),
            ]));
        }
        if let Some(value) = theme.get(&format!("--inset-shadow-{size}")) {
            register_shadow_group(tw_props);
            return Ok(Utility::simple(vec![
                ("--tw-inset-shadow", wrap_inset_shadow_colors(value)),
                ("box-shadow", dialect.box_shadow_chain().to_string()),
            ]));
        }
        if let Some(decls) = shadow_color_decls(
            "--tw-inset-shadow-color",
            "--tw-inset-shadow-alpha",
            size,
            theme,
        ) {
            register_shadow_group(tw_props);
            return Ok(Utility::simple(decls));
        }
        if size.starts_with('[') {
            return Err(unknown(full));
        }
        return Err(Fail::Invalid);
    }

    // inset-ring / inset-ring-<width> / inset-ring-<color>: the inset ring slot.
    // Widths compose into `box-shadow`; a color only assigns `--tw-inset-ring-color`
    // (Tailwind registers no `@property` group for the color form). No negative
    // form. Handled before the position-offset family for the same reason as above.
    if !negative && (positive_base == "inset-ring" || positive_base.starts_with("inset-ring-")) {
        let rest = positive_base.strip_prefix("inset-ring").unwrap_or("");
        let rest = rest.strip_prefix('-').unwrap_or(rest);
        if rest.is_empty() || (!rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit())) {
            let width = if rest.is_empty() { "1" } else { rest };
            register_shadow_group(tw_props);
            return Ok(Utility::simple(vec![
                (
                    "--tw-inset-ring-shadow",
                    format!("inset 0 0 0 {width}px var(--tw-inset-ring-color, currentcolor)"),
                ),
                ("box-shadow", dialect.box_shadow_chain().to_string()),
            ]));
        }
        if let Some(decls) = color_prop_decls("--tw-inset-ring-color", rest, theme) {
            return Ok(Utility::simple(decls));
        }
        if rest.starts_with('[') {
            return Err(unknown(full));
        }
        return Err(Fail::Invalid);
    }

    // contain-*: CSS containment. The simple keywords (`none`/`content`/`strict`)
    // set `contain` directly; the composable slots (`size`/`inline-size`/`layout`/
    // `paint`/`style`) each write a `--tw-contain-*` var plus the composed `contain`
    // shorthand. No negative form and no `/modifier`.
    if positive_base == "contain" || positive_base.starts_with("contain-") {
        if negative {
            return Err(Fail::Invalid);
        }
        let rest = positive_base.strip_prefix("contain-").unwrap_or("");
        if rest.is_empty() || rest.contains('/') {
            return Err(Fail::Invalid);
        }
        match rest {
            "none" => return Ok(Utility::simple(vec![("contain", "none".to_string())])),
            "content" => return Ok(Utility::simple(vec![("contain", "content".to_string())])),
            "strict" => return Ok(Utility::simple(vec![("contain", "strict".to_string())])),
            _ => {}
        }
        let slot = match rest {
            "size" => Some(("--tw-contain-size", "size")),
            "inline-size" => Some(("--tw-contain-size", "inline-size")),
            "layout" => Some(("--tw-contain-layout", "layout")),
            "paint" => Some(("--tw-contain-paint", "paint")),
            "style" => Some(("--tw-contain-style", "style")),
            _ => None,
        };
        if let Some((var, value)) = slot {
            for p in [
                TwProp::ContainSize,
                TwProp::ContainLayout,
                TwProp::ContainPaint,
                TwProp::ContainStyle,
            ] {
                tw_props.insert(p);
            }
            return Ok(Utility::simple(vec![
                (var, value.to_string()),
                (
                    "contain",
                    "var(--tw-contain-size,) var(--tw-contain-layout,) var(--tw-contain-paint,) var(--tw-contain-style,)".to_string(),
                ),
            ]));
        }
        if rest.starts_with('[') {
            return Err(unknown(full));
        }
        return Err(Fail::Invalid);
    }

    // Position offsets: inset/inset-x/inset-y/top/right/bottom/left.
    // Longer prefixes first so `inset-y-0` is not consumed by `inset`.
    let position_families: [(&str, &str, u16); 11] = [
        ("inset-x", "inset-inline", 31),
        ("inset-y", "inset-block", 32),
        ("inset-bs", "inset-block-start", 37),
        ("inset-be", "inset-block-end", 38),
        ("inset-s", "inset-inline-start", 39),
        ("inset-e", "inset-inline-end", 40),
        ("inset", "inset", 30),
        ("top", "top", 33),
        ("right", "right", 34),
        ("bottom", "bottom", 35),
        ("left", "left", 36),
    ];
    for (prefix, property, rank) in position_families {
        if let Some(value) = strip_family(positive_base, prefix) {
            let resolved = offset_value(value, negative).ok_or(Fail::Invalid)?;
            return Ok(Utility::ranked(vec![(property, resolved)], rank));
        }
    }

    // z-index: numbers, `auto`, and arbitrary values (`z-[100]`).
    if let Some(value) = strip_family(positive_base, "z") {
        // `z-auto` -> `z-index:auto`; it has no negative form (`-z-auto` is
        // rejected by Tailwind and generates nothing).
        if value == "auto" {
            if negative {
                return Err(Fail::Invalid);
            }
            return Ok(Utility::simple(vec![("z-index", "auto".to_string())]));
        }
        if value.bytes().all(|b| b.is_ascii_digit()) && !value.is_empty() {
            // Negatives compile to `calc(<n> * -1)`, matching Tailwind.
            let z = if negative {
                format!("calc({value} * -1)")
            } else {
                value.to_string()
            };
            return Ok(Utility::simple(vec![("z-index", z)]));
        }
        if let Some(inner) = arbitrary_value(value) {
            let z = if negative {
                format!("calc({inner} * -1)")
            } else {
                inner
            };
            return Ok(Utility::simple(vec![("z-index", z)]));
        }
        // Anything else (`z-unknown`, `z-123.5`, `z--1`, a stray `/modifier`) is a
        // value Tailwind rejects outright -> generates nothing.
        return Err(Fail::Invalid);
    }

    // flex / basis / grow / shrink / order / grid-cols / grid-rows / grid-flow /
    // auto-cols / auto-rows / col / row families (some accept a leading `-`).
    if let Some(utility) = flex_grid_utility(positive_base, full, negative, theme)? {
        return Ok(utility);
    }

    // Outline family (outline / outline-<width> / outline-none / outline-hidden /
    // outline-<style> / outline-offset-<n> / outline-<color>). Handled before the
    // generic negative gate because `outline-offset-*` accepts a leading `-`.
    if positive_base == "outline" || positive_base.starts_with("outline-") {
        return outline_utility(positive_base, full, negative, theme, tw_props);
    }

    // The background family (`bg-*`) has no negative form; a leading `-` is
    // rejected by Tailwind (it generates nothing, not a hard error).
    if negative && (positive_base == "bg" || positive_base.starts_with("bg-")) {
        return Err(Fail::Invalid);
    }

    // border / divide / ring / rounded reject a leading `-` outright: Tailwind
    // generates nothing (not a hard error).
    if negative
        && (positive_base == "border"
            || positive_base.starts_with("border-")
            || positive_base == "divide"
            || positive_base.starts_with("divide-")
            || positive_base == "ring"
            || positive_base.starts_with("ring-")
            || positive_base == "rounded"
            || positive_base.starts_with("rounded-"))
    {
        return Err(Fail::Invalid);
    }

    // Typography / interactivity families. `underline-offset` accepts a negative
    // numeric value (`-underline-offset-4` -> `calc(4px * -1)`); every other
    // family in this group rejects a leading `-` (Tailwind generates nothing).
    if negative {
        if let Some(rest) = positive_base.strip_prefix("underline-offset-") {
            if !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()) {
                return Ok(Utility::simple(vec![(
                    "text-underline-offset",
                    format!("calc({rest}px * -1)"),
                )]));
            }
            return Err(Fail::Invalid);
        }
        if is_typography_interactivity_family(positive_base) {
            return Err(Fail::Invalid);
        }
    }

    // Sizing (`w`/`h`/`min-*`/`max-*`/`size`) and `aspect` have no negative form;
    // a leading `-` is rejected by Tailwind (it generates nothing).
    if negative
        && ([
            "w-",
            "h-",
            "min-w-",
            "min-h-",
            "max-w-",
            "max-h-",
            "size-",
            "min-block-",
            "max-block-",
            "min-inline-",
            "max-inline-",
        ]
        .iter()
        .any(|p| positive_base.starts_with(p))
            || positive_base == "aspect"
            || positive_base.starts_with("aspect-"))
    {
        return Err(Fail::Invalid);
    }

    // The filter / backdrop-filter chains and `opacity` have no negative form; a
    // leading `-` is rejected by Tailwind (it generates nothing, not an error).
    if negative
        && (matches!(
            positive_base,
            "filter" | "filter-none" | "backdrop-filter" | "backdrop-filter-none"
        ) || positive_base.starts_with("opacity-"))
    {
        return Err(Fail::Invalid);
    }

    if negative {
        // Gradient color stops reject a leading `-` outright (Tailwind generates
        // nothing rather than erroring).
        if ["from-", "via-", "to-"]
            .iter()
            .any(|p| positive_base.starts_with(p))
        {
            return Err(Fail::Invalid);
        }
        // The transition families have no negative form; a leading `-` is
        // rejected by Tailwind (it generates nothing rather than erroring).
        if positive_base == "transition"
            || positive_base.starts_with("transition-")
            || positive_base == "duration"
            || positive_base.starts_with("duration-")
            || positive_base == "delay"
            || positive_base.starts_with("delay-")
            || positive_base == "ease"
            || positive_base.starts_with("ease-")
        {
            return Err(Fail::Invalid);
        }
        // These families have no negative form; a leading `-` generates nothing
        // (not a hard error).
        if positive_base.starts_with("shadow")
            || positive_base.starts_with("clear")
            || positive_base.starts_with("float")
            || positive_base.starts_with("scheme")
            || positive_base.starts_with("contain")
            || positive_base.starts_with("backface")
            || positive_base.starts_with("forced-color")
        {
            return Err(Fail::Invalid);
        }
        // The font-variant-numeric family has no negative form; a leading `-`
        // generates nothing (not a hard error, even though `tabular-nums` is a
        // recognized utility root).
        if matches!(
            positive_base,
            "ordinal"
                | "slashed-zero"
                | "lining-nums"
                | "oldstyle-nums"
                | "proportional-nums"
                | "tabular-nums"
                | "diagonal-fractions"
                | "stacked-fractions"
                | "normal-nums"
        ) {
            return Err(Fail::Invalid);
        }
        // Padding and gap have no negative form; a leading `-` is rejected by
        // Tailwind (it generates nothing, not a hard error).
        if positive_base == "gap"
            || positive_base.starts_with("gap-")
            || ["p-", "px-", "py-", "pt-", "pr-", "pb-", "pl-"]
                .iter()
                .any(|p| positive_base.starts_with(p))
        {
            return Err(Fail::Invalid);
        }
        // Negative margins and negative `space-*` values fall through below; other
        // negatives are unknown.
        if !(positive_base.starts_with('m') || positive_base.starts_with("space-")) {
            return Err(unknown(full));
        }
    }

    // Sizing: w/h/min-w/min-h/max-w/max-h/size.
    if let Some(utility) = sizing_utility(positive_base, full, theme)? {
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

    // space-x-* / space-y-*: reverse-aware adjacent-sibling margin utilities. The
    // `reverse` member sets the reverse var to `1`; a value member writes the
    // reverse var (`0`) and the two calc-composed margins. A `0` value folds to a
    // plain `0` (matching Tailwind's `calc(0 * …)` simplification). Negatives are
    // valid (negative spacing); `space-*-reverse` rejects a leading `-`.
    //
    // The two dialects put the margin on OPPOSITE edges, because they select
    // opposite children: v4 styles every child but the last and leans on the
    // trailing edge, v3 styles every child but the first and leans on the leading
    // one. Each edge below is `(property, carries_the_bare_reverse_var)`; the
    // other edge carries `1 - reverse`. Compiling a v3 app the v4 way put
    // `space-y-4`'s 16px on `margin-bottom` where v3 puts it on `margin-top` (and
    // left the first child un-nudged instead of the last).
    let (space_x_edges, space_y_edges) = match dialect {
        Dialect::V3 => (
            [("margin-right", true), ("margin-left", false)],
            [("margin-top", false), ("margin-bottom", true)],
        ),
        Dialect::V4 => (
            [("margin-inline-start", true), ("margin-inline-end", false)],
            [("margin-block-start", true), ("margin-block-end", false)],
        ),
    };
    for (axis, edges, reverse_var, reverse_prop) in [
        (
            'x',
            space_x_edges,
            "--tw-space-x-reverse",
            TwProp::SpaceXReverse,
        ),
        (
            'y',
            space_y_edges,
            "--tw-space-y-reverse",
            TwProp::SpaceYReverse,
        ),
    ] {
        if positive_base == format!("space-{axis}") {
            return Err(Fail::Invalid);
        }
        if let Some(suffix) = positive_base.strip_prefix(&format!("space-{axis}-")) {
            // A `/modifier` invalidates the token.
            if suffix.contains('/') {
                return Err(Fail::Invalid);
            }
            if suffix == "reverse" {
                if negative {
                    return Err(Fail::Invalid);
                }
                tw_props.insert(reverse_prop);
                return Ok(Utility {
                    selector: SelectorKind::SpaceChildren,
                    decls: vec![(reverse_var.to_string(), "1".to_string())],
                    rank: 100,
                });
            }
            let value = if suffix == "px" {
                Some(if negative {
                    "-1px".to_string()
                } else {
                    "1px".to_string()
                })
            } else if let Some(inner) = arbitrary_value(suffix) {
                Some(if negative {
                    format!("calc({inner} * -1)")
                } else {
                    inner
                })
            } else {
                spacing_value(suffix, negative)
            };
            let value = value.ok_or_else(|| unknown(full))?;
            tw_props.insert(reverse_prop);
            // v4 folds a `0` value to a plain `0`; v3 normalizes it to `0px` and
            // keeps the calc, so `space-y-0` still reads as a length.
            let value = if value == "0" && dialect == Dialect::V3 {
                "0px".to_string()
            } else {
                value
            };
            let reversed = if value == "0" {
                "0".to_string()
            } else {
                format!("calc({value} * var({reverse_var}))")
            };
            let normal = if value == "0" {
                "0".to_string()
            } else {
                format!("calc({value} * calc(1 - var({reverse_var})))")
            };
            let mut decls = vec![(reverse_var.to_string(), "0".to_string())];
            for (prop, carries_reverse) in edges {
                let val = if carries_reverse {
                    reversed.clone()
                } else {
                    normal.clone()
                };
                decls.push((prop.to_string(), val));
            }
            return Ok(Utility {
                selector: SelectorKind::SpaceChildren,
                decls,
                rank: 100,
            });
        }
    }

    // rounded / rounded-<size> / rounded-<side>(-<size>): border-radius from
    // the theme radius scale, whole-box or per side/corner.
    if base == "rounded" {
        return Ok(Utility::ranked(
            vec![("border-radius", "0.25rem".to_string())],
            45,
        ));
    }
    if let Some(rest) = base.strip_prefix("rounded-") {
        let (side, size) = match rest.split_once('-') {
            Some((side, size)) if rounded_side_rank(side).is_some() => (side, size),
            None if rounded_side_rank(rest).is_some() => (rest, ""),
            _ => ("", rest),
        };
        let value = radius_value(size, theme).ok_or_else(|| unknown(full))?;
        if side.is_empty() {
            return Ok(Utility::ranked(vec![("border-radius", value)], 45));
        }
        let rank = rounded_side_rank(side).unwrap();
        let decls = rounded_side_properties(side)
            .iter()
            .map(|prop| (*prop, value.clone()))
            .collect::<Vec<_>>();
        return Ok(Utility::ranked(decls, rank));
    }

    // cursor-<keyword>.
    if let Some(kw) = base.strip_prefix("cursor-") {
        if is_cursor_keyword(kw) {
            return Ok(Utility::simple(vec![("cursor", kw.to_string())]));
        }
        return Err(unknown(full));
    }

    // opacity-<n>: a percentage. The bare value is a multiple of 0.25, emitted as
    // `<n>%`; an arbitrary `[…]` value passes through raw (an explicit `type:`
    // hint always resolves, otherwise the value must read as a `<number>` or
    // `<percentage>` — a bare `var()`/function is rejected). Anything else
    // (non-multiples, negatives) is a value Tailwind rejects outright.
    if let Some(value) = base.strip_prefix("opacity-") {
        if is_spacing_multiplier(value) {
            return Ok(Utility::simple(vec![("opacity", format!("{value}%"))]));
        }
        if let Some(inner) = arbitrary_value(value) {
            if let Some((_, rest)) = split_data_type_hint(&inner) {
                return Ok(Utility::simple(vec![("opacity", rest.to_string())]));
            }
            if is_number_or_percentage(&inner) {
                return Ok(Utility::simple(vec![("opacity", inner)]));
            }
        }
        return Err(Fail::Invalid);
    }

    // line-clamp-<n> / line-clamp-none.
    if let Some(n) = base.strip_prefix("line-clamp-") {
        if n == "none" {
            return Ok(Utility::simple(vec![
                ("overflow", "visible".to_string()),
                ("display", "block".to_string()),
                ("-webkit-box-orient", "horizontal".to_string()),
                ("-webkit-line-clamp", "unset".to_string()),
            ]));
        }
        if n.bytes().all(|b| b.is_ascii_digit()) && !n.is_empty() {
            return Ok(Utility::simple(vec![
                ("overflow", "hidden".to_string()),
                ("display", "-webkit-box".to_string()),
                ("-webkit-box-orient", "vertical".to_string()),
                ("-webkit-line-clamp", n.to_string()),
            ]));
        }
        if n.starts_with('[') {
            return Err(unknown(full));
        }
        return Err(Fail::Invalid);
    }

    // underline-offset-<n> / underline-offset-auto. The bare `underline-offset`
    // and any non-numeric/non-`auto` value are rejected (Tailwind emits nothing);
    // a `/<modifier>` is likewise rejected. Negatives are handled in the group's
    // negative gate above.
    if base == "underline-offset" {
        return Err(Fail::Invalid);
    }
    if let Some(rest) = base.strip_prefix("underline-offset-") {
        let (core, has_mod) = match rest.split_once('/') {
            Some((c, _)) => (c, true),
            None => (rest, false),
        };
        if has_mod {
            return Err(Fail::Invalid);
        }
        if core == "auto" {
            return Ok(Utility::simple(vec![(
                "text-underline-offset",
                "auto".to_string(),
            )]));
        }
        if !core.is_empty() && core.bytes().all(|b| b.is_ascii_digit()) {
            return Ok(Utility::simple(vec![(
                "text-underline-offset",
                format!("{core}px"),
            )]));
        }
        if core.starts_with('[') {
            return Err(unknown(full));
        }
        return Err(Fail::Invalid);
    }

    // leading-<value>: --tw-leading + line-height. v3 has no `--tw-leading` slot
    // (it was introduced in v4 so `leading-*` beats a `text-<size>`'s own
    // line-height regardless of source order); a v3 entry emits the bare
    // `line-height` and lets the cascade decide, exactly as v3 does.
    if let Some(rest) = base.strip_prefix("leading-") {
        let value = if rest == "none" {
            Some("1".to_string())
        } else if theme.contains(&format!("--leading-{rest}")) {
            Some(format!("var(--leading-{rest})"))
        } else {
            spacing_value(rest, false)
        };
        let value = value.ok_or_else(|| unknown(full))?;
        if dialect == Dialect::V3 {
            return Ok(Utility::simple(vec![("line-height", value)]));
        }
        tw_props.insert(TwProp::Leading);
        return Ok(Utility::simple(vec![
            ("--tw-leading", value.clone()),
            ("line-height", value),
        ]));
    }

    // tracking-<name>: --tw-tracking + letter-spacing. Takes no `/<modifier>`.
    if let Some(rest) = base.strip_prefix("tracking-") {
        let (core, has_mod) = match rest.split_once('/') {
            Some((c, _)) => (c, true),
            None => (rest, false),
        };
        let var = format!("--tracking-{core}");
        if theme.contains(&var) {
            if has_mod {
                return Err(Fail::Invalid);
            }
            tw_props.insert(TwProp::Tracking);
            return Ok(Utility::simple(vec![
                ("--tw-tracking", format!("var({var})")),
                ("letter-spacing", format!("var({var})")),
            ]));
        }
        if core.starts_with('[') {
            return Err(unknown(full));
        }
        return Err(Fail::Invalid);
    }

    // font-variant-numeric family: each of ordinal / slashed-zero / lining-nums /
    // oldstyle-nums / proportional-nums / tabular-nums / diagonal-fractions /
    // stacked-fractions sets its own `--tw-*` slot and composes
    // `font-variant-numeric` from the shared var chain; `normal-nums` resets it to
    // `normal`. None has a negative form (a leading `-` generates nothing).
    if let Some(var) = match positive_base {
        "ordinal" => Some("--tw-ordinal"),
        "slashed-zero" => Some("--tw-slashed-zero"),
        "lining-nums" | "oldstyle-nums" => Some("--tw-numeric-figure"),
        "proportional-nums" | "tabular-nums" => Some("--tw-numeric-spacing"),
        "diagonal-fractions" | "stacked-fractions" => Some("--tw-numeric-fraction"),
        _ => None,
    } {
        for p in [
            TwProp::Ordinal,
            TwProp::SlashedZero,
            TwProp::NumericFigure,
            TwProp::NumericSpacing,
            TwProp::NumericFraction,
        ] {
            tw_props.insert(p);
        }
        return Ok(Utility::simple(vec![
            (var, positive_base.to_string()),
            (
                "font-variant-numeric",
                "var(--tw-ordinal,) var(--tw-slashed-zero,) var(--tw-numeric-figure,) var(--tw-numeric-spacing,) var(--tw-numeric-fraction,)".to_string(),
            ),
        ]));
    }
    if positive_base == "normal-nums" {
        return Ok(Utility::simple(vec![(
            "font-variant-numeric",
            "normal".to_string(),
        )]));
    }

    // shadow / shadow-<size> / shadow-none / shadow-[…]: box-shadow layers from
    // the theme shadow scale (bare `shadow` is the scale's `sm` entry, per the
    // reference) or an arbitrary shadow list, colors wrapped for
    // `--tw-shadow-color`.
    if base == "shadow" || base.starts_with("shadow-") {
        let size = base.strip_prefix("shadow-").unwrap_or("sm");
        let shadow = if size == "none" {
            Some("0 0 #0000".to_string())
        } else if let Some(value) = theme.get(&format!("--shadow-{size}")) {
            Some(wrap_shadow_colors(value))
        } else {
            arbitrary_value(size).map(|inner| wrap_shadow_colors(&inner))
        };
        if let Some(shadow) = shadow {
            register_shadow_group(tw_props);
            return Ok(Utility::simple(vec![
                ("--tw-shadow", shadow),
                ("box-shadow", dialect.box_shadow_chain().to_string()),
            ]));
        }
        // shadow-<color>: assign `--tw-shadow-color` (composed with the shadow
        // alpha), plus the static sRGB fallback line.
        if let Some(decls) =
            shadow_color_decls("--tw-shadow-color", "--tw-shadow-alpha", size, theme)
        {
            register_shadow_group(tw_props);
            return Ok(Utility::simple(decls));
        }
        return Err(unknown(full));
    }

    // drop-shadow / drop-shadow-<size>: filter drop-shadow layers from the
    // theme scale. A single-layer theme value keeps the `var(--drop-shadow-*)`
    // reference for the plain fallback; multi-layer values (the bare default)
    // inline each layer, exactly as the reference compiles them.
    if base == "drop-shadow" || base.starts_with("drop-shadow-") {
        let size = base.strip_prefix("drop-shadow-").unwrap_or("");
        let (sized, plain) = if size.is_empty() {
            // The bare `drop-shadow` default, verbatim from Tailwind v4 (the
            // published theme has no bare `--drop-shadow` token).
            let value = "0 1px 2px rgb(0 0 0 / 0.1), 0 1px 1px rgb(0 0 0 / 0.06)";
            (
                drop_shadow_layers(value, true),
                drop_shadow_layers(value, false),
            )
        } else if let Some(value) = theme.get(&format!("--drop-shadow-{size}")) {
            let plain = if value.contains(',') {
                drop_shadow_layers(value, false)
            } else {
                format!("drop-shadow(var(--drop-shadow-{size}))")
            };
            (drop_shadow_layers(value, true), plain)
        } else {
            return Err(unknown(full));
        };
        register_filter_group(tw_props);
        return Ok(Utility::simple(vec![
            ("--tw-drop-shadow-size", sized),
            ("--tw-drop-shadow", plain),
            ("filter", FILTER_CHAIN.to_string()),
        ]));
    }

    // backdrop-blur / backdrop-blur-<size> (the backdrop-filter blur family;
    // bare uses the scale's `sm` entry inlined, per the reference).
    if base == "backdrop-blur" || base.starts_with("backdrop-blur-") {
        let size = base.strip_prefix("backdrop-blur-").unwrap_or("");
        let blur = if size.is_empty() {
            let value = theme.get("--blur-sm").ok_or_else(|| unknown(full))?;
            format!("blur({value})")
        } else if size == "none" {
            // `--tw-backdrop-blur: ;` — the empty (whitespace) value Tailwind
            // uses to clear the composed filter slot.
            " ".to_string()
        } else if theme.contains(&format!("--blur-{size}")) {
            format!("blur(var(--blur-{size}))")
        } else if let Some(inner) = arbitrary_value(size) {
            format!("blur({inner})")
        } else {
            return Err(unknown(full));
        };
        register_backdrop_group(tw_props);
        return Ok(Utility::simple(vec![
            ("--tw-backdrop-blur", blur),
            ("-webkit-backdrop-filter", BACKDROP_FILTER_CHAIN.to_string()),
            ("backdrop-filter", BACKDROP_FILTER_CHAIN.to_string()),
        ]));
    }

    // blur / blur-<size> / blur-none / blur-[…]: the `filter` blur family
    // (mirrors `backdrop-blur`, composing into the `--tw-blur` filter slot).
    if base == "blur" || base.starts_with("blur-") {
        let size = base.strip_prefix("blur-").unwrap_or("");
        let blur = if size.is_empty() {
            let value = theme.get("--blur-sm").ok_or_else(|| unknown(full))?;
            format!("blur({value})")
        } else if size == "none" {
            // `--tw-blur: ;` — the empty (whitespace) value clearing the slot.
            " ".to_string()
        } else if theme.contains(&format!("--blur-{size}")) {
            format!("blur(var(--blur-{size}))")
        } else if let Some(inner) = arbitrary_value(size) {
            format!("blur({inner})")
        } else {
            return Err(unknown(full));
        };
        register_filter_group(tw_props);
        return Ok(Utility::simple(vec![
            ("--tw-blur", blur),
            ("filter", FILTER_CHAIN.to_string()),
        ]));
    }

    // filter / filter-none: the composed filter chain (or its reset). The chain
    // form registers every `--tw-*` filter slot; the reset registers nothing.
    if base == "filter" {
        register_filter_group(tw_props);
        return Ok(Utility::simple(vec![("filter", FILTER_CHAIN.to_string())]));
    }
    if base == "filter-none" {
        return Ok(Utility::simple(vec![("filter", "none".to_string())]));
    }

    // backdrop-filter / backdrop-filter-none: the composed backdrop-filter chain
    // (prefixed for WebKit) or its reset.
    if base == "backdrop-filter" {
        register_backdrop_group(tw_props);
        return Ok(Utility::simple(vec![
            ("-webkit-backdrop-filter", BACKDROP_FILTER_CHAIN.to_string()),
            ("backdrop-filter", BACKDROP_FILTER_CHAIN.to_string()),
        ]));
    }
    if base == "backdrop-filter-none" {
        return Ok(Utility::simple(vec![
            ("-webkit-backdrop-filter", "none".to_string()),
            ("backdrop-filter", "none".to_string()),
        ]));
    }

    // ring / ring-<n> / ring-inset / ring-offset-<n> / ring-<color>.
    if base == "ring" || base.starts_with("ring-") {
        return ring_utility(base, full, theme, tw_props, dialect);
    }

    // divide-x / divide-y (widths + reverse), divide-<style>, divide-<color>.
    if base == "divide" || base.starts_with("divide-") {
        return divide_utility(base, full, theme, tw_props, dialect);
    }

    // (outline utilities are handled above, before the negative gate.)

    // transition families (leading `-` already rejected by the negative gate).
    if let Some(decls) = transition_utility(positive_base) {
        return Ok(Utility::simple(decls));
    }
    if let Some(rest) = positive_base.strip_prefix("transition-") {
        // Arbitrary transition properties are an engine gap; any other
        // unknown suffix (`transition-color`) is a value Tailwind resolves
        // against nothing and drops.
        if rest.starts_with('[') {
            return Err(unknown(full));
        }
        return Err(Fail::Invalid);
    }

    // duration-<ms> / duration-[…]: --tw-duration + transition-duration.
    // `duration-initial` only resets the custom property (no transition-duration).
    if let Some(value) = positive_base.strip_prefix("duration-") {
        if value == "initial" {
            tw_props.insert(TwProp::Duration);
            return Ok(Utility::simple(vec![(
                "--tw-duration",
                "initial".to_string(),
            )]));
        }
        let resolved = if value.bytes().all(|b| b.is_ascii_digit()) && !value.is_empty() {
            format!("{value}ms")
        } else if value.starts_with('[') {
            // A malformed arbitrary value is a genuine engine gap.
            arbitrary_value(value).ok_or_else(|| unknown(full))?
        } else {
            // Any other token resolves against nothing — Tailwind drops it.
            return Err(Fail::Invalid);
        };
        tw_props.insert(TwProp::Duration);
        return Ok(Utility::simple(vec![
            ("--tw-duration", resolved.clone()),
            ("transition-duration", resolved),
        ]));
    }

    // delay-<ms> / delay-[…]: transition-delay (no custom property).
    if let Some(value) = positive_base.strip_prefix("delay-") {
        let resolved = if value.bytes().all(|b| b.is_ascii_digit()) && !value.is_empty() {
            format!("{value}ms")
        } else if value.starts_with('[') {
            arbitrary_value(value).ok_or_else(|| unknown(full))?
        } else {
            return Err(Fail::Invalid);
        };
        return Ok(Utility::simple(vec![("transition-delay", resolved)]));
    }

    // ease-<name> / ease-[…]: --tw-ease + transition-timing-function.
    // `ease-linear` is a literal; `ease-initial` only resets the custom property.
    if let Some(name) = positive_base.strip_prefix("ease-") {
        if name == "initial" {
            tw_props.insert(TwProp::Ease);
            return Ok(Utility::simple(vec![("--tw-ease", "initial".to_string())]));
        }
        let resolved = if name == "linear" {
            "linear".to_string()
        } else if theme.contains(&format!("--ease-{name}")) {
            format!("var(--ease-{name})")
        } else if name.starts_with('[') {
            arbitrary_value(name).ok_or_else(|| unknown(full))?
        } else {
            return Err(Fail::Invalid);
        };
        tw_props.insert(TwProp::Ease);
        return Ok(Utility::simple(vec![
            ("--tw-ease", resolved.clone()),
            ("transition-timing-function", resolved),
        ]));
    }

    // vertical-align.
    if let Some(align) = base.strip_prefix("align-") {
        if matches!(
            align,
            "baseline" | "top" | "middle" | "bottom" | "text-top" | "text-bottom" | "sub" | "super"
        ) {
            return Ok(Utility::simple(vec![("vertical-align", align.to_string())]));
        }
        return Err(unknown(full));
    }

    // aspect-ratio: square/video/auto, fractions, arbitrary. Any other value
    // resolves against nothing — Tailwind generates no rule for it.
    if let Some(value) = base.strip_prefix("aspect-") {
        let resolved = match value {
            "auto" => Some("auto".to_string()),
            "square" => Some("1 / 1".to_string()),
            "video" if theme.contains("--aspect-video") => Some("var(--aspect-video)".to_string()),
            _ => parse_fraction(value)
                .map(|(n, d)| format!("{n}/{d}"))
                .or_else(|| arbitrary_value(value)),
        };
        let Some(resolved) = resolved else {
            return Err(Fail::Invalid);
        };
        return Ok(Utility::simple(vec![("aspect-ratio", resolved)]));
    }

    // animate-<name>: the theme's --animate-* scale (the matching @keyframes is
    // emitted with the stylesheet). A name with no theme token resolves against
    // nothing — Tailwind generates no rule for it.
    if let Some(name) = base.strip_prefix("animate-") {
        if name == "none" {
            return Ok(Utility::simple(vec![("animation", "none".to_string())]));
        }
        let var = format!("--animate-{name}");
        if theme.contains(&var) {
            return Ok(Utility::simple(vec![("animation", format!("var({var})"))]));
        }
        if let Some(inner) = arbitrary_value(name) {
            return Ok(Utility::simple(vec![("animation", inner)]));
        }
        return Err(Fail::Invalid);
    }

    // Gradient color stops: from-*/via-*/to-* colors or stop positions. A value
    // that is neither resolves against nothing — Tailwind generates no rule.
    for (family, rank) in [("from", 102u16), ("via", 103), ("to", 104)] {
        let Some(value) = strip_family(base, family) else {
            continue;
        };
        // Stop positions: `from-10%`. Tailwind only accepts a bare non-negative
        // integer percentage here; decimals (`25.5%`) and negatives are rejected.
        if let Some(pct) = value.strip_suffix('%')
            && !pct.is_empty()
            && pct.bytes().all(|b| b.is_ascii_digit())
        {
            register_gradient_group(tw_props, dialect);
            return Ok(Utility::ranked(
                vec![(gradient_position_property(family), format!("{pct}%"))],
                rank + 3,
            ));
        }
        // v3 composes `--tw-gradient-stops` inline out of `<color> <position>`
        // pairs and has no `--tw-gradient-via`/`--tw-gradient-via-stops` at all.
        if dialect == Dialect::V3 {
            let Some(decls) = v3_gradient_stop_decls(family, value, theme) else {
                return Err(Fail::Invalid);
            };
            register_gradient_group(tw_props, dialect);
            return Ok(Utility::ranked(decls, rank));
        }
        let color_prop = match family {
            "from" => "--tw-gradient-from",
            "via" => "--tw-gradient-via",
            _ => "--tw-gradient-to",
        };
        let Some(color_decls) = gradient_color_decls(color_prop, value, theme) else {
            return Err(Fail::Invalid);
        };
        register_gradient_group(tw_props, dialect);
        let mut decls: Vec<(&str, String)> = color_decls;
        match family {
            "from" => decls.push(("--tw-gradient-stops", GRADIENT_STOPS.to_string())),
            "via" => {
                decls.push(("--tw-gradient-via-stops", GRADIENT_VIA_STOPS.to_string()));
                decls.push((
                    "--tw-gradient-stops",
                    "var(--tw-gradient-via-stops)".to_string(),
                ));
            }
            _ => decls.push(("--tw-gradient-stops", GRADIENT_STOPS.to_string())),
        }
        return Ok(Utility::ranked(decls, rank));
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

    // accent-<color> (with optional `/<pct>` opacity modifier).
    if let Some(rest) = base.strip_prefix("accent-") {
        if let Some(decls) = color_prop_decls("accent-color", rest, theme) {
            return Ok(Utility::simple(decls));
        }
        if rest.starts_with('[') {
            return Err(unknown(full));
        }
        return Err(Fail::Invalid);
    }

    // placeholder-<color> (with optional `/<pct>` opacity modifier): the input
    // placeholder text color, baked onto the `::placeholder` pseudo-element.
    if let Some(rest) = base.strip_prefix("placeholder-") {
        if let Some(decls) = color_prop_decls("color", rest, theme) {
            return Ok(Utility {
                selector: SelectorKind::ClassPseudoElement("::placeholder".to_string()),
                decls: decls
                    .into_iter()
                    .map(|(prop, value)| (prop.to_string(), value))
                    .collect(),
                rank: 100,
            });
        }
        if rest.starts_with('[') {
            return Err(unknown(full));
        }
        return Err(Fail::Invalid);
    }

    // caret-<color> (with optional `/<pct>` opacity modifier).
    if let Some(rest) = base.strip_prefix("caret-") {
        if let Some(decls) = color_prop_decls("caret-color", rest, theme) {
            return Ok(Utility::simple(decls));
        }
        if rest.starts_with('[') {
            return Err(unknown(full));
        }
        return Err(Fail::Invalid);
    }

    // fill-<color> / fill-none (SVG fill; color takes a `/<pct>` modifier).
    if let Some(rest) = base.strip_prefix("fill-") {
        if let Some(decls) = color_prop_decls("fill", rest, theme) {
            return Ok(Utility::simple(decls));
        }
        if rest == "none" {
            return Ok(Utility::simple(vec![("fill", "none".to_string())]));
        }
        if rest.starts_with('[') {
            return Err(unknown(full));
        }
        return Err(Fail::Invalid);
    }

    // stroke-<color> / stroke-none / stroke-<width> (SVG stroke).
    if let Some(rest) = base.strip_prefix("stroke-") {
        if let Some(decls) = color_prop_decls("stroke", rest, theme) {
            return Ok(Utility::simple(decls));
        }
        if rest == "none" {
            return Ok(Utility::simple(vec![("stroke", "none".to_string())]));
        }
        if !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()) {
            return Ok(Utility::simple(vec![("stroke-width", rest.to_string())]));
        }
        if rest.starts_with('[') {
            return Err(unknown(full));
        }
        return Err(Fail::Invalid);
    }

    // decoration-<style>/<thickness>/<color>: text-decoration-* utilities.
    if let Some(rest) = base.strip_prefix("decoration-") {
        let (core, has_mod) = match rest.split_once('/') {
            Some((c, _)) => (c, true),
            None => (rest, false),
        };
        if matches!(core, "solid" | "double" | "dotted" | "dashed" | "wavy") {
            if has_mod {
                return Err(Fail::Invalid);
            }
            return Ok(Utility::simple(vec![(
                "text-decoration-style",
                core.to_string(),
            )]));
        }
        if matches!(core, "auto" | "from-font") {
            if has_mod {
                return Err(Fail::Invalid);
            }
            return Ok(Utility::simple(vec![(
                "text-decoration-thickness",
                core.to_string(),
            )]));
        }
        if !core.is_empty() && core.bytes().all(|b| b.is_ascii_digit()) {
            if has_mod {
                return Err(Fail::Invalid);
            }
            return Ok(Utility::simple(vec![(
                "text-decoration-thickness",
                format!("{core}px"),
            )]));
        }
        if let Some(decls) = color_prop_decls("text-decoration-color", rest, theme) {
            return Ok(Utility::simple(decls));
        }
        if rest.starts_with('[') {
            return Err(unknown(full));
        }
        return Err(Fail::Invalid);
    }

    // list-*: the valid `list-inside`/`list-outside`/`list-item`/`list-disc`/
    // `list-decimal`/`list-none`/`list-image-none` keywords are handled in
    // `keyword_utility`; any other `list-*` value Tailwind rejects (an arbitrary
    // `list-[…]` is an engine gap).
    if let Some(rest) = base.strip_prefix("list-") {
        if rest.starts_with('[') {
            return Err(unknown(full));
        }
        return Err(Fail::Invalid);
    }

    // touch-action: the simple keywords plus the composed `pan-*`/`pinch-zoom`
    // slots. Takes no `/<modifier>`.
    if let Some(rest) = base.strip_prefix("touch-") {
        let (core, has_mod) = match rest.split_once('/') {
            Some((c, _)) => (c, true),
            None => (rest, false),
        };
        if has_mod {
            return Err(Fail::Invalid);
        }
        if matches!(core, "auto" | "none" | "manipulation") {
            return Ok(Utility::simple(vec![("touch-action", core.to_string())]));
        }
        let slot = match core {
            "pan-x" => Some(("--tw-pan-x", "pan-x")),
            "pan-left" => Some(("--tw-pan-x", "pan-left")),
            "pan-right" => Some(("--tw-pan-x", "pan-right")),
            "pan-y" => Some(("--tw-pan-y", "pan-y")),
            "pan-up" => Some(("--tw-pan-y", "pan-up")),
            "pan-down" => Some(("--tw-pan-y", "pan-down")),
            "pinch-zoom" => Some(("--tw-pinch-zoom", "pinch-zoom")),
            _ => None,
        };
        if let Some((var, value)) = slot {
            for prop in [TwProp::PanX, TwProp::PanY, TwProp::PinchZoom] {
                tw_props.insert(prop);
            }
            return Ok(Utility::simple(vec![
                (var, value.to_string()),
                (
                    "touch-action",
                    "var(--tw-pan-x,) var(--tw-pan-y,) var(--tw-pinch-zoom,)".to_string(),
                ),
            ]));
        }
        if rest.starts_with('[') {
            return Err(unknown(full));
        }
        return Err(Fail::Invalid);
    }

    // text-shadow / text-shadow-<size> / text-shadow-none: the theme text-shadow
    // scale, colors wrapped for `--tw-text-shadow-color`.
    if base == "text-shadow" || base.starts_with("text-shadow-") {
        let size = base.strip_prefix("text-shadow-").unwrap_or("");
        if size == "none" {
            return Ok(Utility::simple(vec![("text-shadow", "none".to_string())]));
        }
        if let Some(value) = theme.get(&format!("--text-shadow-{size}")) {
            tw_props.insert(TwProp::TextShadowColor);
            tw_props.insert(TwProp::TextShadowAlpha);
            return Ok(Utility::simple(vec![(
                "text-shadow",
                wrap_text_shadow_colors(value),
            )]));
        }
        return Err(unknown(full));
    }

    // bg-gradient-to-<dir> (v3 spelling) / bg-linear-to-<dir>: the linear
    // gradient position plus the composed background-image.
    if let Some(dir) = base
        .strip_prefix("bg-gradient-to-")
        .or_else(|| base.strip_prefix("bg-linear-to-"))
    {
        let position = match dir {
            "t" => "to top",
            "tr" => "to top right",
            "r" => "to right",
            "br" => "to bottom right",
            "b" => "to bottom",
            "bl" => "to bottom left",
            "l" => "to left",
            "tl" => "to top left",
            _ => return Err(unknown(full)),
        };
        register_gradient_group(tw_props, dialect);
        // v3 writes the direction straight into `background-image` and interpolates
        // in sRGB; v4 routes it through `--tw-gradient-position` and interpolates
        // `in oklab`. Compiling a v3 app the v4 way made `bg-gradient-to-r` compute
        // `linear-gradient(to right in oklab, …)`, a visibly different ramp.
        if dialect == Dialect::V3 {
            return Ok(Utility::ranked(
                vec![(
                    "background-image",
                    format!("linear-gradient({position}, var(--tw-gradient-stops))"),
                )],
                101,
            ));
        }
        return Ok(Utility::ranked(
            vec![
                ("--tw-gradient-position", format!("{position} in oklab")),
                (
                    "background-image",
                    "linear-gradient(var(--tw-gradient-stops))".to_string(),
                ),
            ],
            101,
        ));
    }

    // bg-clip-<box> (background-clip) / bg-origin-<box> (background-origin).
    // Neither accepts an opacity `/<modifier>` (Tailwind rejects `bg-clip-border/foo`).
    for (prefix, prop) in [
        ("bg-clip-", "background-clip"),
        ("bg-origin-", "background-origin"),
    ] {
        if let Some(rest) = base.strip_prefix(prefix) {
            let (core, has_mod) = match rest.split_once('/') {
                Some((c, _)) => (c, true),
                None => (rest, false),
            };
            let value = match (prefix, core) {
                ("bg-clip-", "border") => Some("border-box"),
                ("bg-clip-", "content") => Some("content-box"),
                ("bg-clip-", "padding") => Some("padding-box"),
                ("bg-clip-", "text") => Some("text"),
                ("bg-origin-", "border") => Some("border-box"),
                ("bg-origin-", "content") => Some("content-box"),
                ("bg-origin-", "padding") => Some("padding-box"),
                _ => None,
            };
            if let Some(value) = value {
                if has_mod {
                    return Err(Fail::Invalid);
                }
                return Ok(Utility::simple(vec![(prop, value.to_string())]));
            }
            // Not a valid box keyword: fall through to the color / unknown path.
        }
    }

    // bg-<color> (with optional `/<pct>` opacity modifier). The `/<pct>` modifier
    // additionally emits the static `color-mix(in srgb, …)` fallback Tailwind ships
    // for pre-`oklab` browsers (via `color_prop_decls`).
    if let Some(color) = base.strip_prefix("bg-") {
        if let Some(decls) = color_prop_decls("background-color", color, theme) {
            return Ok(Utility::simple(decls));
        }
        // A `family-shade`-shaped unknown color (`bg-plaid-500`) is a likely typo —
        // a hard error naming the token. A bare word (`bg-clip`, `bg-position`,
        // `bg-unknown`) or a color with an invalid `/<modifier>` (`bg-current/half`,
        // `bg-red-500/half`) is simply not a utility: Tailwind emits nothing.
        if color.starts_with('[') || (color.contains('-') && !color.contains('/')) {
            return Err(unknown(full));
        }
        return Err(Fail::Invalid);
    }

    // font-<weight> / font-<family> / font-stretch-<value>.
    if let Some(rest) = base.strip_prefix("font-") {
        // font-stretch-<percentage|keyword>.
        if rest == "stretch" {
            return Err(Fail::Invalid);
        }
        if let Some(stretch) = rest.strip_prefix("stretch-") {
            let (core, has_mod) = match stretch.split_once('/') {
                Some((c, _)) => (c, true),
                None => (stretch, false),
            };
            if has_mod {
                return Err(Fail::Invalid);
            }
            if let Some(value) = font_stretch_value(core) {
                return Ok(Utility::simple(vec![("font-stretch", value)]));
            }
            if core.starts_with('[') {
                return Err(unknown(full));
            }
            return Err(Fail::Invalid);
        }
        let (core, has_mod) = match rest.split_once('/') {
            Some((c, _)) => (c, true),
            None => (rest, false),
        };
        if is_font_weight(core) {
            if has_mod {
                return Err(Fail::Invalid);
            }
            tw_props.insert(TwProp::FontWeight);
            let var = format!("--font-weight-{core}");
            return Ok(Utility::simple(vec![
                ("--tw-font-weight", format!("var({var})")),
                ("font-weight", format!("var({var})")),
            ]));
        }
        if matches!(core, "sans" | "serif" | "mono") {
            if has_mod {
                return Err(Fail::Invalid);
            }
            return Ok(Utility::simple(vec![(
                "font-family",
                format!("var(--font-{core})"),
            )]));
        }
        if rest.starts_with('[') {
            return Err(unknown(full));
        }
        return Err(Fail::Invalid);
    }

    // text-<align>/<overflow>/<size>/<color>/[arbitrary]. Alignment, overflow, and
    // size take no `/<modifier>`; a color does (its `/<pct>` opacity).
    if let Some(rest) = base.strip_prefix("text-") {
        let (core, has_mod) = match rest.split_once('/') {
            Some((c, _)) => (c, true),
            None => (rest, false),
        };
        if matches!(
            core,
            "left" | "center" | "right" | "justify" | "start" | "end"
        ) {
            if has_mod {
                return Err(Fail::Invalid);
            }
            return Ok(Utility::simple(vec![("text-align", core.to_string())]));
        }
        if core == "ellipsis" || core == "clip" {
            if has_mod {
                return Err(Fail::Invalid);
            }
            return Ok(Utility::simple(vec![("text-overflow", core.to_string())]));
        }
        if is_text_size(core) {
            if has_mod {
                return Err(Fail::Invalid);
            }
            let size = format!("--text-{core}");
            let leading = format!("--text-{core}--line-height");
            // v3 has no `--tw-leading` indirection: `text-4xl` writes its own
            // line-height and a `leading-*` only wins when it comes later in the
            // sheet. v3's plugin order puts `fontSize` BEFORE `lineHeight`, so
            // `text-3xl leading-snug` is 1.375 there — reproduced by ranking the
            // size ahead of the default bucket `leading-*` sits in. A v3 config may
            // also give a size NO line-height at all (`fontSize: {'5xl':'2.5rem'}`),
            // in which case v3 emits only `font-size` and the line-height inherits.
            if dialect == Dialect::V3 {
                let mut decls = vec![("font-size", format!("var({size})"))];
                if theme.contains(&leading) {
                    decls.push(("line-height", format!("var({leading})")));
                }
                return Ok(Utility::ranked(decls, TEXT_SIZE_RANK_V3));
            }
            return Ok(Utility::simple(vec![
                ("font-size", format!("var({size})")),
                ("line-height", format!("var(--tw-leading, var({leading}))")),
            ]));
        }
        if rest.starts_with('[') {
            // `text-[color:…]` is a color (with optional `/<pct>`); a bare
            // arbitrary value is a size.
            if let Some(decls) = color_prop_decls("color", rest, theme) {
                return Ok(Utility::simple(decls));
            }
            let inner = arbitrary_value(rest).ok_or_else(|| unknown(full))?;
            return Ok(Utility::simple(vec![("font-size", inner)]));
        }
        // text-<color> (with optional `/<pct>` opacity modifier).
        if let Some(decls) = color_prop_decls("color", rest, theme) {
            return Ok(Utility::simple(decls));
        }
        // An unknown value shaped like a `family-shade` color reference
        // (`text-gray-1000`) is a likely typo — a hard error naming the token.
        // A bare word (`text-trim`) is simply not a utility: Tailwind emits
        // nothing.
        if core.contains('-') || core.starts_with('[') {
            return Err(unknown(full));
        }
        return Err(Fail::Invalid);
    }

    Err(unknown(full))
}

// ---------------------------------------------------------------------------
// The transform family: translate / scale / rotate / skew / transform /
// perspective / transform-origin / perspective-origin. `positive_base` has had
// the negative-marking leading `-` removed; `negative` records whether it was
// present. Returns `None` when the base is not in this family (fall through);
// otherwise `Some(Ok(..))` for a generated rule or `Some(Err(Fail::Invalid))`
// for a token Tailwind rejects and drops (an unknown value, a stray modifier,
// or a `-` on a non-negatable form).
fn transform_family_utility(
    positive_base: &str,
    negative: bool,
    theme: &Theme,
    tw_props: &mut BTreeSet<TwProp>,
) -> Option<Result<Utility, Fail>> {
    // perspective-origin-* must be checked before the generic perspective-*.
    if let Some(rest) = positive_base.strip_prefix("perspective-origin-") {
        return Some(origin_like("perspective-origin", rest, negative));
    }
    if positive_base == "perspective" {
        return Some(Err(Fail::Invalid));
    }
    if let Some(rest) = positive_base.strip_prefix("perspective-") {
        return Some(perspective_utility(rest, negative, theme));
    }

    // transform-origin.
    if let Some(rest) = positive_base.strip_prefix("origin-") {
        return Some(origin_like("transform-origin", rest, negative));
    }

    // transform / transform-*.
    if positive_base == "transform" {
        if negative {
            return Some(Err(Fail::Invalid));
        }
        register_rotate_skew(tw_props);
        return Some(Ok(Utility::simple(vec![(
            "transform",
            TRANSFORM_CHAIN.to_string(),
        )])));
    }
    if let Some(rest) = positive_base.strip_prefix("transform-") {
        return Some(transform_variant_utility(rest, negative, tw_props));
    }

    // rotate / rotate-x/y/z.
    if positive_base == "rotate" {
        return Some(Err(Fail::Invalid));
    }
    if let Some(rest) = positive_base.strip_prefix("rotate-") {
        return Some(rotate_utility(rest, negative, tw_props));
    }

    // skew / skew-x/y.
    if positive_base == "skew" {
        return Some(Err(Fail::Invalid));
    }
    if let Some(rest) = positive_base.strip_prefix("skew-") {
        return Some(skew_utility(rest, negative, tw_props));
    }

    // scale / scale-x/y/z / scale-3d.
    if positive_base == "scale" {
        return Some(Err(Fail::Invalid));
    }
    if let Some(rest) = positive_base.strip_prefix("scale-") {
        return Some(scale_utility(rest, negative, tw_props));
    }

    // translate / translate-x/y/z / translate-3d.
    if positive_base == "translate" {
        return Some(Err(Fail::Invalid));
    }
    if let Some(rest) = positive_base.strip_prefix("translate-") {
        return Some(translate_utility(rest, negative, tw_props));
    }

    None
}

fn register_rotate_skew(tw_props: &mut BTreeSet<TwProp>) {
    for prop in [
        TwProp::RotateX,
        TwProp::RotateY,
        TwProp::RotateZ,
        TwProp::SkewX,
        TwProp::SkewY,
    ] {
        tw_props.insert(prop);
    }
}

fn register_scale(tw_props: &mut BTreeSet<TwProp>) {
    for prop in [TwProp::ScaleX, TwProp::ScaleY, TwProp::ScaleZ] {
        tw_props.insert(prop);
    }
}

fn register_translate(tw_props: &mut BTreeSet<TwProp>) {
    for prop in [TwProp::TranslateX, TwProp::TranslateY, TwProp::TranslateZ] {
        tw_props.insert(prop);
    }
}

/// A bare integer angle (`45` -> `45deg`) or a bracketed arbitrary value,
/// negated as `calc(<v> * -1)`. Decimals and any other form are rejected —
/// Tailwind only accepts integers or arbitrary values here.
fn angle_value(value: &str, negative: bool) -> Option<String> {
    let raw = if !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()) {
        format!("{value}deg")
    } else {
        arbitrary_value(value)?
    };
    Some(if negative {
        format!("calc({raw} * -1)")
    } else {
        raw
    })
}

/// A per-axis scale value: a bare integer percentage (`50` -> `50%`) or a
/// bracketed arbitrary value, negated as `calc(<v> * -1)`.
fn scale_axis_value(value: &str, negative: bool) -> Option<String> {
    let raw = if !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()) {
        format!("{value}%")
    } else {
        arbitrary_value(value)?
    };
    Some(if negative {
        format!("calc({raw} * -1)")
    } else {
        raw
    })
}

/// The `translate-z` value type: spacing-scale numbers, `px`, or an arbitrary
/// value — but NOT fractions or `full` (those are percentage forms the Z axis
/// does not accept).
fn translate_z_value(value: &str, negative: bool) -> Option<String> {
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

/// The shared `transform-origin` / `perspective-origin` keyword table (both
/// families map the same position words to the same values).
fn transform_origin_keyword(rest: &str) -> Option<&'static str> {
    Some(match rest {
        "center" => "center",
        "top" => "top",
        "top-right" => "100% 0",
        "right" => "100%",
        "bottom-right" => "100% 100%",
        "bottom" => "bottom",
        "bottom-left" => "0 100%",
        "left" => "0",
        "top-left" => "0 0",
        _ => return None,
    })
}

/// `origin-*` (transform-origin) and `perspective-origin-*` share one shape:
/// a position keyword or an arbitrary value; never negatable.
fn origin_like(property: &'static str, rest: &str, negative: bool) -> Result<Utility, Fail> {
    if negative {
        return Err(Fail::Invalid);
    }
    let value = transform_origin_keyword(rest)
        .map(|s| s.to_string())
        .or_else(|| arbitrary_value(rest))
        .ok_or(Fail::Invalid)?;
    Ok(Utility::simple(vec![(property, value)]))
}

/// `perspective-<token>`: `none`, a theme `--perspective-*` name, or an
/// arbitrary value. Bare numbers resolve against nothing. Never negatable.
fn perspective_utility(rest: &str, negative: bool, theme: &Theme) -> Result<Utility, Fail> {
    if negative {
        return Err(Fail::Invalid);
    }
    let value = if rest == "none" {
        "none".to_string()
    } else if theme.contains(&format!("--perspective-{rest}")) {
        format!("var(--perspective-{rest})")
    } else if let Some(inner) = arbitrary_value(rest) {
        inner
    } else {
        return Err(Fail::Invalid);
    };
    Ok(Utility::simple(vec![("perspective", value)]))
}

/// The static `transform-*` variants: `none`/`cpu`/`gpu` (the `transform`
/// property), `flat`/`3d` (transform-style), the `transform-box` words, and an
/// arbitrary value. Never negatable. Only bare `transform`, `transform-cpu`,
/// `transform-gpu`, and `transform-[…]` reference the rotate/skew slots — but
/// Tailwind registers the `@property` set only for bare `transform` and the
/// arbitrary form, not for `cpu`/`gpu`.
fn transform_variant_utility(
    rest: &str,
    negative: bool,
    tw_props: &mut BTreeSet<TwProp>,
) -> Result<Utility, Fail> {
    if negative {
        return Err(Fail::Invalid);
    }
    let decls: Vec<(&str, String)> = match rest {
        "none" => vec![("transform", "none".to_string())],
        "cpu" => vec![("transform", TRANSFORM_CHAIN.to_string())],
        "gpu" => vec![("transform", format!("translateZ(0) {TRANSFORM_CHAIN}"))],
        "flat" => vec![("transform-style", "flat".to_string())],
        "3d" => vec![("transform-style", "preserve-3d".to_string())],
        "content" => vec![("transform-box", "content-box".to_string())],
        "border" => vec![("transform-box", "border-box".to_string())],
        "fill" => vec![("transform-box", "fill-box".to_string())],
        "stroke" => vec![("transform-box", "stroke-box".to_string())],
        "view" => vec![("transform-box", "view-box".to_string())],
        _ => {
            let inner = arbitrary_value(rest).ok_or(Fail::Invalid)?;
            register_rotate_skew(tw_props);
            return Ok(Utility::simple(vec![("transform", inner)]));
        }
    };
    Ok(Utility::simple(decls))
}

/// `rotate-*`: bare `rotate-<angle>` sets the `rotate` property directly;
/// `rotate-x/y/z-<angle>` set a `--tw-rotate-*` slot wrapped in `rotate{X,Y,Z}()`
/// and feed the transform chain; `rotate-none` clears the property.
fn rotate_utility(
    rest: &str,
    negative: bool,
    tw_props: &mut BTreeSet<TwProp>,
) -> Result<Utility, Fail> {
    if rest == "none" {
        if negative {
            return Err(Fail::Invalid);
        }
        return Ok(Utility::simple(vec![("rotate", "none".to_string())]));
    }
    if let Some((axis @ ("x" | "y" | "z"), value)) = rest.split_once('-') {
        let angle = angle_value(value, negative).ok_or(Fail::Invalid)?;
        let func = match axis {
            "x" => "rotateX",
            "y" => "rotateY",
            _ => "rotateZ",
        };
        register_rotate_skew(tw_props);
        return Ok(Utility {
            selector: SelectorKind::Class,
            decls: vec![
                (format!("--tw-rotate-{axis}"), format!("{func}({angle})")),
                ("transform".to_string(), TRANSFORM_CHAIN.to_string()),
            ],
            rank: 100,
        });
    }
    let angle = angle_value(rest, negative).ok_or(Fail::Invalid)?;
    Ok(Utility::simple(vec![("rotate", angle)]))
}

/// `skew-*`: bare `skew-<angle>` sets both axes; `skew-x/y-<angle>` sets one.
/// Each slot is wrapped in `skew{X,Y}()` and feeds the transform chain.
fn skew_utility(
    rest: &str,
    negative: bool,
    tw_props: &mut BTreeSet<TwProp>,
) -> Result<Utility, Fail> {
    if let Some((axis @ ("x" | "y"), value)) = rest.split_once('-') {
        let angle = angle_value(value, negative).ok_or(Fail::Invalid)?;
        let func = if axis == "x" { "skewX" } else { "skewY" };
        register_rotate_skew(tw_props);
        return Ok(Utility {
            selector: SelectorKind::Class,
            decls: vec![
                (format!("--tw-skew-{axis}"), format!("{func}({angle})")),
                ("transform".to_string(), TRANSFORM_CHAIN.to_string()),
            ],
            rank: 100,
        });
    }
    let angle = angle_value(rest, negative).ok_or(Fail::Invalid)?;
    register_rotate_skew(tw_props);
    Ok(Utility {
        selector: SelectorKind::Class,
        decls: vec![
            ("--tw-skew-x".to_string(), format!("skewX({angle})")),
            ("--tw-skew-y".to_string(), format!("skewY({angle})")),
            ("transform".to_string(), TRANSFORM_CHAIN.to_string()),
        ],
        rank: 100,
    })
}

/// `scale-*`: `scale-3d` engages all three axes; `scale-x/y/z-<n>` set one slot;
/// bare `scale-<int>` sets all three `--tw-scale-*` slots (the shorthand still
/// lists only x/y); bare `scale-[…]` assigns the `scale` property directly.
fn scale_utility(
    rest: &str,
    negative: bool,
    tw_props: &mut BTreeSet<TwProp>,
) -> Result<Utility, Fail> {
    if rest == "3d" {
        if negative {
            return Err(Fail::Invalid);
        }
        register_scale(tw_props);
        return Ok(Utility::simple(vec![(
            "scale",
            "var(--tw-scale-x) var(--tw-scale-y) var(--tw-scale-z)".to_string(),
        )]));
    }
    if let Some((axis @ ("x" | "y" | "z"), value)) = rest.split_once('-') {
        let resolved = scale_axis_value(value, negative).ok_or(Fail::Invalid)?;
        register_scale(tw_props);
        let shorthand = if axis == "z" {
            "var(--tw-scale-x) var(--tw-scale-y) var(--tw-scale-z)"
        } else {
            "var(--tw-scale-x) var(--tw-scale-y)"
        };
        return Ok(Utility {
            selector: SelectorKind::Class,
            decls: vec![
                (format!("--tw-scale-{axis}"), resolved),
                ("scale".to_string(), shorthand.to_string()),
            ],
            rank: 100,
        });
    }
    // Bare arbitrary values assign `scale` directly (no `--tw-*` slots).
    if let Some(inner) = arbitrary_value(rest) {
        let value = if negative {
            format!("calc({inner} * -1)")
        } else {
            inner
        };
        return Ok(Utility::simple(vec![("scale", value)]));
    }
    // Bare integers set all three axes.
    if !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()) {
        let value = if negative {
            format!("calc({rest}% * -1)")
        } else {
            format!("{rest}%")
        };
        register_scale(tw_props);
        return Ok(Utility::simple(vec![
            ("--tw-scale-x", value.clone()),
            ("--tw-scale-y", value.clone()),
            ("--tw-scale-z", value),
            ("scale", "var(--tw-scale-x) var(--tw-scale-y)".to_string()),
        ]));
    }
    Err(Fail::Invalid)
}

/// `translate-*`: `translate-none` clears; `translate-3d` engages all three
/// axes; `translate-x/y-<v>` and bare `translate-<v>` take fractions / spacing /
/// `px` / `full` / arbitrary; `translate-z-<v>` takes spacing / `px` / arbitrary
/// only.
fn translate_utility(
    rest: &str,
    negative: bool,
    tw_props: &mut BTreeSet<TwProp>,
) -> Result<Utility, Fail> {
    if rest == "none" {
        if negative {
            return Err(Fail::Invalid);
        }
        return Ok(Utility::simple(vec![("translate", "none".to_string())]));
    }
    if rest == "3d" {
        if negative {
            return Err(Fail::Invalid);
        }
        register_translate(tw_props);
        return Ok(Utility::simple(vec![(
            "translate",
            "var(--tw-translate-x) var(--tw-translate-y) var(--tw-translate-z)".to_string(),
        )]));
    }
    if let Some((axis @ ("x" | "y" | "z"), value)) = rest.split_once('-') {
        let resolved = if axis == "z" {
            translate_z_value(value, negative)
        } else {
            translate_value(value, negative)
        }
        .ok_or(Fail::Invalid)?;
        register_translate(tw_props);
        let shorthand = if axis == "z" {
            "var(--tw-translate-x) var(--tw-translate-y) var(--tw-translate-z)"
        } else {
            "var(--tw-translate-x) var(--tw-translate-y)"
        };
        return Ok(Utility {
            selector: SelectorKind::Class,
            decls: vec![
                (format!("--tw-translate-{axis}"), resolved),
                ("translate".to_string(), shorthand.to_string()),
            ],
            rank: 100,
        });
    }
    let resolved = translate_value(rest, negative).ok_or(Fail::Invalid)?;
    register_translate(tw_props);
    Ok(Utility::simple(vec![
        ("--tw-translate-x", resolved.clone()),
        ("--tw-translate-y", resolved),
        (
            "translate",
            "var(--tw-translate-x) var(--tw-translate-y)".to_string(),
        ),
    ]))
}

/// Single-keyword utilities.
/// A CSS blend-mode keyword (shared by `mix-blend-*` and `bg-blend-*`). `plus`
/// enables the compositing-only `plus-darker`/`plus-lighter` modes that only
/// `mix-blend-mode` accepts.
fn blend_mode(value: &str, plus: bool) -> Option<&'static str> {
    Some(match value {
        "normal" => "normal",
        "multiply" => "multiply",
        "screen" => "screen",
        "overlay" => "overlay",
        "darken" => "darken",
        "lighten" => "lighten",
        "color-dodge" => "color-dodge",
        "color-burn" => "color-burn",
        "hard-light" => "hard-light",
        "soft-light" => "soft-light",
        "difference" => "difference",
        "exclusion" => "exclusion",
        "hue" => "hue",
        "saturation" => "saturation",
        "color" => "color",
        "luminosity" => "luminosity",
        "plus-darker" if plus => "plus-darker",
        "plus-lighter" if plus => "plus-lighter",
        _ => return None,
    })
}

/// The `object-position` keyword for an `object-<value>` value, or `None` if the
/// value is not a position keyword (the caller then tries `object-fit`).
fn object_position(value: &str) -> Option<&'static str> {
    Some(match value {
        "top" => "top",
        "bottom" => "bottom",
        "left" => "left",
        "right" => "right",
        "center" => "center",
        "left-top" | "top-left" => "left top",
        "left-bottom" | "bottom-left" => "left bottom",
        "right-top" | "top-right" => "right top",
        "right-bottom" | "bottom-right" => "right bottom",
        _ => return None,
    })
}

/// Static keyword utilities in the scroll/overflow/snap/blend/break/box group,
/// keyed by the base (no negative prefix, no `/modifier`). The `bool` is whether
/// the utility registers the `--tw-scroll-snap-strictness` property.
fn scroll_overflow_static(base: &str) -> Option<(Vec<(&'static str, String)>, bool)> {
    let decls: Vec<(&'static str, String)> = match base {
        "isolate" => vec![("isolation", "isolate".into())],
        "box-border" => vec![("box-sizing", "border-box".into())],
        "box-content" => vec![("box-sizing", "content-box".into())],
        "box-decoration-clone" => vec![
            ("-webkit-box-decoration-break", "clone".into()),
            ("box-decoration-break", "clone".into()),
        ],
        "box-decoration-slice" => vec![
            ("-webkit-box-decoration-break", "slice".into()),
            ("box-decoration-break", "slice".into()),
        ],
        "break-all" => vec![("word-break", "break-all".into())],
        "break-keep" => vec![("word-break", "keep-all".into())],
        "break-normal" => {
            vec![
                ("overflow-wrap", "normal".into()),
                ("word-break", "normal".into()),
            ]
        }
        "break-words" => vec![("overflow-wrap", "break-word".into())],
        "overflow-auto" => vec![("overflow", "auto".into())],
        "overflow-hidden" => vec![("overflow", "hidden".into())],
        "overflow-clip" => vec![("overflow", "clip".into())],
        "overflow-visible" => vec![("overflow", "visible".into())],
        "overflow-scroll" => vec![("overflow", "scroll".into())],
        "overflow-x-auto" => vec![("overflow-x", "auto".into())],
        "overflow-x-hidden" => vec![("overflow-x", "hidden".into())],
        "overflow-x-clip" => vec![("overflow-x", "clip".into())],
        "overflow-x-visible" => vec![("overflow-x", "visible".into())],
        "overflow-x-scroll" => vec![("overflow-x", "scroll".into())],
        "overflow-y-auto" => vec![("overflow-y", "auto".into())],
        "overflow-y-hidden" => vec![("overflow-y", "hidden".into())],
        "overflow-y-clip" => vec![("overflow-y", "clip".into())],
        "overflow-y-visible" => vec![("overflow-y", "visible".into())],
        "overflow-y-scroll" => vec![("overflow-y", "scroll".into())],
        "overscroll-auto" => vec![("overscroll-behavior", "auto".into())],
        "overscroll-contain" => vec![("overscroll-behavior", "contain".into())],
        "overscroll-none" => vec![("overscroll-behavior", "none".into())],
        "overscroll-x-auto" => vec![("overscroll-behavior-x", "auto".into())],
        "overscroll-x-contain" => vec![("overscroll-behavior-x", "contain".into())],
        "overscroll-x-none" => vec![("overscroll-behavior-x", "none".into())],
        "overscroll-y-auto" => vec![("overscroll-behavior-y", "auto".into())],
        "overscroll-y-contain" => vec![("overscroll-behavior-y", "contain".into())],
        "overscroll-y-none" => vec![("overscroll-behavior-y", "none".into())],
        "scroll-auto" => vec![("scroll-behavior", "auto".into())],
        "scroll-smooth" => vec![("scroll-behavior", "smooth".into())],
        "snap-none" => vec![("scroll-snap-type", "none".into())],
        "snap-x" => {
            return Some((
                vec![(
                    "scroll-snap-type",
                    "x var(--tw-scroll-snap-strictness)".into(),
                )],
                true,
            ));
        }
        "snap-y" => {
            return Some((
                vec![(
                    "scroll-snap-type",
                    "y var(--tw-scroll-snap-strictness)".into(),
                )],
                true,
            ));
        }
        "snap-both" => {
            return Some((
                vec![(
                    "scroll-snap-type",
                    "both var(--tw-scroll-snap-strictness)".into(),
                )],
                true,
            ));
        }
        "snap-mandatory" => {
            return Some((
                vec![("--tw-scroll-snap-strictness", "mandatory".into())],
                true,
            ));
        }
        "snap-proximity" => {
            return Some((
                vec![("--tw-scroll-snap-strictness", "proximity".into())],
                true,
            ));
        }
        "snap-start" => vec![("scroll-snap-align", "start".into())],
        "snap-end" => vec![("scroll-snap-align", "end".into())],
        "snap-center" => vec![("scroll-snap-align", "center".into())],
        "snap-align-none" => vec![("scroll-snap-align", "none".into())],
        "snap-normal" => vec![("scroll-snap-stop", "normal".into())],
        "snap-always" => vec![("scroll-snap-stop", "always".into())],
        "scrollbar-auto" => vec![("scrollbar-width", "auto".into())],
        "scrollbar-thin" => vec![("scrollbar-width", "thin".into())],
        "scrollbar-none" => vec![("scrollbar-width", "none".into())],
        "scrollbar-gutter-auto" => vec![("scrollbar-gutter", "auto".into())],
        "scrollbar-gutter-stable" => vec![("scrollbar-gutter", "stable".into())],
        "scrollbar-gutter-both" => vec![("scrollbar-gutter", "stable both-edges".into())],
        _ => return None,
    };
    Some((decls, false))
}

/// The `--tw-scrollbar-thumb`/`--tw-scrollbar-track` declaration(s) for a color
/// token (with optional `/<pct>` modifier). A plain token yields one `var(...)`
/// declaration; a modifier yields the `color-mix(in oklab, …)` value plus the
/// static sRGB fallback Tailwind emits for browsers without `oklab`.
fn scrollbar_color_decls(var: &str, token: &str, theme: &Theme) -> Option<Vec<(String, String)>> {
    let resolved = color_value(token, theme)?;
    let (color_token, modifier) = split_color_modifier(token);
    match modifier {
        None => Some(vec![(var.to_string(), resolved)]),
        Some(pct) => {
            // The static fallback resolves the theme color to its literal value.
            let literal = match color_token {
                "transparent" => "transparent".to_string(),
                "current" => "currentcolor".to_string(),
                "inherit" => "inherit".to_string(),
                _ if color_token.starts_with('[') => arbitrary_value(color_token)?,
                _ => theme.get(&format!("--color-{color_token}"))?.to_string(),
            };
            Some(vec![
                (var.to_string(), resolved),
                (
                    var.to_string(),
                    format!("color-mix(in srgb, {literal} {pct}%, transparent)"),
                ),
            ])
        }
    }
}

/// The scroll/overflow/snap/blend/break/box/columns/object family. Returns `None`
/// when `base` is not one of these families (fall through to the rest of the
/// generator), `Some(Err(Fail::Invalid))` for a token in the family that Tailwind
/// itself rejects (it generates nothing), and `Some(Ok(..))` for a match.
fn scroll_overflow_utility(
    base: &str,
    theme: &Theme,
    tw_props: &mut BTreeSet<TwProp>,
) -> Option<Result<Utility, Fail>> {
    let (negative, b) = match base.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, base),
    };

    // Color scrollbar utilities accept a `/<pct>` modifier, so handle them before
    // splitting the generic modifier off.
    for (prefix, var) in [
        ("scrollbar-thumb-", "--tw-scrollbar-thumb"),
        ("scrollbar-track-", "--tw-scrollbar-track"),
    ] {
        if let Some(token) = b.strip_prefix(prefix) {
            if negative {
                return Some(Err(Fail::Invalid));
            }
            match scrollbar_color_decls(var, token, theme) {
                Some(mut decls) => {
                    tw_props.insert(TwProp::ScrollbarThumb);
                    tw_props.insert(TwProp::ScrollbarTrack);
                    decls.push((
                        "scrollbar-color".to_string(),
                        "var(--tw-scrollbar-thumb) var(--tw-scrollbar-track)".to_string(),
                    ));
                    return Some(Ok(Utility {
                        selector: SelectorKind::Class,
                        decls,
                        rank: 100,
                    }));
                }
                None => return Some(Err(Fail::Invalid)),
            }
        }
    }
    if matches!(
        b,
        "scrollbar-thumb" | "scrollbar-track" | "scrollbar" | "scrollbar-gutter"
    ) {
        return Some(Err(Fail::Invalid));
    }

    let (core, has_mod) = match b.split_once('/') {
        Some((c, m)) => (c, Some(m)),
        None => (b, None),
    };

    // Static keyword utilities.
    if let Some((decls, snap_prop)) = scroll_overflow_static(core) {
        if negative || has_mod.is_some() {
            return Some(Err(Fail::Invalid));
        }
        if snap_prop {
            tw_props.insert(TwProp::ScrollSnapStrictness);
        }
        return Some(Ok(Utility::simple(decls)));
    }

    // Bare family names (and their `x`/`y` axes) generate nothing.
    if matches!(
        core,
        "box"
            | "box-decoration"
            | "break-after"
            | "break-before"
            | "break-inside"
            | "overflow-x"
            | "overflow-y"
            | "overscroll-x"
            | "overscroll-y"
            | "mix-blend"
            | "bg-blend"
            | "snap"
    ) {
        return Some(Err(Fail::Invalid));
    }

    // mix-blend-* / bg-blend-*.
    if let Some(v) = core.strip_prefix("mix-blend-") {
        if negative || has_mod.is_some() {
            return Some(Err(Fail::Invalid));
        }
        return Some(match blend_mode(v, true) {
            Some(m) => Ok(Utility::simple(vec![("mix-blend-mode", m.to_string())])),
            None => Err(Fail::Invalid),
        });
    }
    if let Some(v) = core.strip_prefix("bg-blend-") {
        if negative || has_mod.is_some() {
            return Some(Err(Fail::Invalid));
        }
        return Some(match blend_mode(v, false) {
            Some(m) => Ok(Utility::simple(vec![(
                "background-blend-mode",
                m.to_string(),
            )])),
            None => Err(Fail::Invalid),
        });
    }

    // break-after / break-before / break-inside value families.
    for (prefix, property) in [
        ("break-after-", "break-after"),
        ("break-before-", "break-before"),
    ] {
        if let Some(v) = core.strip_prefix(prefix) {
            if negative || has_mod.is_some() {
                return Some(Err(Fail::Invalid));
            }
            let ok = matches!(
                v,
                "auto" | "avoid" | "all" | "avoid-page" | "page" | "left" | "right" | "column"
            );
            return Some(if ok {
                Ok(Utility::simple(vec![(property, v.to_string())]))
            } else {
                Err(Fail::Invalid)
            });
        }
    }
    if let Some(v) = core.strip_prefix("break-inside-") {
        if negative || has_mod.is_some() {
            return Some(Err(Fail::Invalid));
        }
        let ok = matches!(v, "auto" | "avoid" | "avoid-page" | "avoid-column");
        return Some(if ok {
            Ok(Utility::simple(vec![("break-inside", v.to_string())]))
        } else {
            Err(Fail::Invalid)
        });
    }

    // object-* : object-fit keyword, object-position keyword, or arbitrary value.
    if let Some(v) = core.strip_prefix("object-") {
        if negative || has_mod.is_some() {
            return Some(Err(Fail::Invalid));
        }
        if matches!(v, "contain" | "cover" | "fill" | "none" | "scale-down") {
            return Some(Ok(Utility::simple(vec![("object-fit", v.to_string())])));
        }
        if let Some(pos) = object_position(v) {
            return Some(Ok(Utility::simple(vec![(
                "object-position",
                pos.to_string(),
            )])));
        }
        if let Some(inner) = arbitrary_value(v) {
            return Some(Ok(Utility::simple(vec![("object-position", inner)])));
        }
        return Some(Err(Fail::Invalid));
    }

    // columns-* : a bare number, the container scale, `auto`, or an arbitrary value.
    if core == "columns" {
        return Some(Err(Fail::Invalid));
    }
    if let Some(v) = core.strip_prefix("columns-") {
        if negative || has_mod.is_some() {
            return Some(Err(Fail::Invalid));
        }
        if v == "auto" {
            return Some(Ok(Utility::simple(vec![("columns", "auto".to_string())])));
        }
        if !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()) {
            return Some(Ok(Utility::simple(vec![("columns", v.to_string())])));
        }
        let container = format!("--container-{v}");
        if theme.contains(&container) {
            return Some(Ok(Utility::simple(vec![(
                "columns",
                format!("var({container})"),
            )])));
        }
        if let Some(inner) = arbitrary_value(v) {
            return Some(Ok(Utility::simple(vec![("columns", inner)])));
        }
        return Some(Err(Fail::Invalid));
    }

    // scroll-margin / scroll-padding scale (and reject unknown `scroll-*`).
    if let Some(rest) = core.strip_prefix("scroll-") {
        const SCROLL_SPACING: &[(&str, &str, bool)] = &[
            ("mbs", "scroll-margin-block-start", true),
            ("mbe", "scroll-margin-block-end", true),
            ("mx", "scroll-margin-inline", true),
            ("my", "scroll-margin-block", true),
            ("ms", "scroll-margin-inline-start", true),
            ("me", "scroll-margin-inline-end", true),
            ("mt", "scroll-margin-top", true),
            ("mr", "scroll-margin-right", true),
            ("mb", "scroll-margin-bottom", true),
            ("ml", "scroll-margin-left", true),
            ("m", "scroll-margin", true),
            ("pbs", "scroll-padding-block-start", false),
            ("pbe", "scroll-padding-block-end", false),
            ("px", "scroll-padding-inline", false),
            ("py", "scroll-padding-block", false),
            ("ps", "scroll-padding-inline-start", false),
            ("pe", "scroll-padding-inline-end", false),
            ("pt", "scroll-padding-top", false),
            ("pr", "scroll-padding-right", false),
            ("pb", "scroll-padding-bottom", false),
            ("pl", "scroll-padding-left", false),
            ("p", "scroll-padding", false),
        ];
        for &(prefix, property, is_margin) in SCROLL_SPACING {
            // Bare family name (`scroll-m`) generates nothing.
            if rest == prefix {
                return Some(Err(Fail::Invalid));
            }
            if let Some(step) = rest.strip_prefix(prefix).and_then(|r| r.strip_prefix('-')) {
                if has_mod.is_some() {
                    return Some(Err(Fail::Invalid));
                }
                if negative && !is_margin {
                    return Some(Err(Fail::Invalid));
                }
                if step == "px" {
                    let value = if negative { "-1px" } else { "1px" };
                    return Some(Ok(Utility::simple(vec![(property, value.to_string())])));
                }
                if let Some(inner) = arbitrary_value(step) {
                    let value = if negative {
                        format!("calc({inner} * -1)")
                    } else {
                        inner
                    };
                    return Some(Ok(Utility::simple(vec![(property, value)])));
                }
                return Some(match spacing_value(step, negative) {
                    Some(value) => Ok(Utility::simple(vec![(property, value)])),
                    None => Err(Fail::Invalid),
                });
            }
        }
        // Any other `scroll-*` token is not a utility.
        return Some(Err(Fail::Invalid));
    }

    None
}

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
        "visible" => vec![("visibility", "visible".into())],
        "invisible" => vec![("visibility", "hidden".into())],
        "collapse" => vec![("visibility", "collapse".into())],
        "isolation-auto" => vec![("isolation", "auto".into())],
        "flex-col" => vec![("flex-direction", "column".into())],
        "flex-col-reverse" => vec![("flex-direction", "column-reverse".into())],
        "flex-row" => vec![("flex-direction", "row".into())],
        "flex-row-reverse" => vec![("flex-direction", "row-reverse".into())],
        "flex-wrap" => vec![("flex-wrap", "wrap".into())],
        "flex-wrap-reverse" => vec![("flex-wrap", "wrap-reverse".into())],
        "flex-nowrap" => vec![("flex-wrap", "nowrap".into())],
        // (`flex-1` / `flex-auto` / `flex-initial` / `flex-none` are handled in
        // `flex_grid_utility` alongside the dynamic `flex-<n>` / `flex-<fraction>`
        // forms, so that `flex-1/2` reads as the fraction 1/2, not `flex-1` with a
        // rejected `/2` modifier.)
        // (align/justify/place alignment utilities live in `alignment_utility`.)
        "list-disc" => vec![("list-style-type", "disc".into())],
        "list-decimal" => vec![("list-style-type", "decimal".into())],
        "list-none" => vec![("list-style-type", "none".into())],
        "uppercase" => vec![("text-transform", "uppercase".into())],
        "lowercase" => vec![("text-transform", "lowercase".into())],
        "capitalize" => vec![("text-transform", "capitalize".into())],
        "normal-case" => vec![("text-transform", "none".into())],
        "italic" => vec![("font-style", "italic".into())],
        "not-italic" => vec![("font-style", "normal".into())],
        "antialiased" => vec![
            ("-webkit-font-smoothing", "antialiased".into()),
            ("-moz-osx-font-smoothing", "grayscale".into()),
        ],
        "subpixel-antialiased" => vec![
            ("-webkit-font-smoothing", "auto".into()),
            ("-moz-osx-font-smoothing", "auto".into()),
        ],
        // Border-STYLE keywords set both the modern `--tw-border-style` (which the
        // border-WIDTH utilities read) and `border-style` directly, matching v4.
        "border-solid" => vec![
            ("--tw-border-style", "solid".into()),
            ("border-style", "solid".into()),
        ],
        "border-dashed" => vec![
            ("--tw-border-style", "dashed".into()),
            ("border-style", "dashed".into()),
        ],
        "border-dotted" => vec![
            ("--tw-border-style", "dotted".into()),
            ("border-style", "dotted".into()),
        ],
        "border-double" => vec![
            ("--tw-border-style", "double".into()),
            ("border-style", "double".into()),
        ],
        "border-hidden" => vec![
            ("--tw-border-style", "hidden".into()),
            ("border-style", "hidden".into()),
        ],
        "border-none" => vec![
            ("--tw-border-style", "none".into()),
            ("border-style", "none".into()),
        ],
        "border-collapse" => vec![("border-collapse", "collapse".into())],
        "border-separate" => vec![("border-collapse", "separate".into())],
        "underline" => vec![("text-decoration-line", "underline".into())],
        "overline" => vec![("text-decoration-line", "overline".into())],
        "line-through" => vec![("text-decoration-line", "line-through".into())],
        "no-underline" => vec![("text-decoration-line", "none".into())],
        "whitespace-normal" => vec![("white-space", "normal".into())],
        "whitespace-nowrap" => vec![("white-space", "nowrap".into())],
        "whitespace-pre" => vec![("white-space", "pre".into())],
        "whitespace-pre-line" => vec![("white-space", "pre-line".into())],
        "whitespace-pre-wrap" => vec![("white-space", "pre-wrap".into())],
        "whitespace-break-spaces" => vec![("white-space", "break-spaces".into())],
        "table" => vec![("display", "table".into())],
        "inline-table" => vec![("display", "inline-table".into())],
        "table-auto" => vec![("table-layout", "auto".into())],
        "table-fixed" => vec![("table-layout", "fixed".into())],
        "caption-top" => vec![("caption-side", "top".into())],
        "caption-bottom" => vec![("caption-side", "bottom".into())],
        "table-caption" => vec![("display", "table-caption".into())],
        "table-cell" => vec![("display", "table-cell".into())],
        "table-column" => vec![("display", "table-column".into())],
        "table-column-group" => vec![("display", "table-column-group".into())],
        "table-footer-group" => vec![("display", "table-footer-group".into())],
        "table-header-group" => vec![("display", "table-header-group".into())],
        "table-row" => vec![("display", "table-row".into())],
        "table-row-group" => vec![("display", "table-row-group".into())],
        "flow-root" => vec![("display", "flow-root".into())],
        "select-none" => {
            vec![
                ("-webkit-user-select", "none".into()),
                ("user-select", "none".into()),
            ]
        }
        "select-text" => {
            vec![
                ("-webkit-user-select", "text".into()),
                ("user-select", "text".into()),
            ]
        }
        "select-all" => {
            vec![
                ("-webkit-user-select", "all".into()),
                ("user-select", "all".into()),
            ]
        }
        "select-auto" => {
            vec![
                ("-webkit-user-select", "auto".into()),
                ("user-select", "auto".into()),
            ]
        }
        "pointer-events-none" => vec![("pointer-events", "none".into())],
        "pointer-events-auto" => vec![("pointer-events", "auto".into())],
        // list-style position / marker display / image reset.
        "list-inside" => vec![("list-style-position", "inside".into())],
        "list-outside" => vec![("list-style-position", "outside".into())],
        "list-item" => vec![("display", "list-item".into())],
        "list-image-none" => vec![("list-style-image", "none".into())],
        // text-wrap.
        "text-wrap" => vec![("text-wrap", "wrap".into())],
        "text-nowrap" => vec![("text-wrap", "nowrap".into())],
        "text-balance" => vec![("text-wrap", "balance".into())],
        "text-pretty" => vec![("text-wrap", "pretty".into())],
        // overflow-wrap (`wrap-*`).
        "wrap-normal" => vec![("overflow-wrap", "normal".into())],
        "wrap-break-word" => vec![("overflow-wrap", "break-word".into())],
        "wrap-anywhere" => vec![("overflow-wrap", "anywhere".into())],
        // appearance.
        "appearance-none" => vec![("appearance", "none".into())],
        "appearance-auto" => vec![("appearance", "auto".into())],
        // resize.
        "resize" => vec![("resize", "both".into())],
        "resize-none" => vec![("resize", "none".into())],
        "resize-x" => vec![("resize", "horizontal".into())],
        "resize-y" => vec![("resize", "vertical".into())],
        // field-sizing.
        "field-sizing-content" => vec![("field-sizing", "content".into())],
        "field-sizing-fixed" => vec![("field-sizing", "fixed".into())],
        // hyphens (WebKit-prefixed).
        "hyphens-none" => vec![
            ("-webkit-hyphens", "none".into()),
            ("hyphens", "none".into()),
        ],
        "hyphens-manual" => vec![
            ("-webkit-hyphens", "manual".into()),
            ("hyphens", "manual".into()),
        ],
        "hyphens-auto" => vec![
            ("-webkit-hyphens", "auto".into()),
            ("hyphens", "auto".into()),
        ],
        // will-change.
        "will-change-auto" => vec![("will-change", "auto".into())],
        "will-change-scroll" => vec![("will-change", "scroll-position".into())],
        "will-change-contents" => vec![("will-change", "contents".into())],
        "will-change-transform" => vec![("will-change", "transform".into())],
        // clear (logical `start`/`end` map to `inline-start`/`inline-end`).
        "clear-left" => vec![("clear", "left".into())],
        "clear-right" => vec![("clear", "right".into())],
        "clear-both" => vec![("clear", "both".into())],
        "clear-none" => vec![("clear", "none".into())],
        "clear-start" => vec![("clear", "inline-start".into())],
        "clear-end" => vec![("clear", "inline-end".into())],
        // float (logical `start`/`end`).
        "float-right" => vec![("float", "right".into())],
        "float-left" => vec![("float", "left".into())],
        "float-none" => vec![("float", "none".into())],
        "float-start" => vec![("float", "inline-start".into())],
        "float-end" => vec![("float", "inline-end".into())],
        // backface-visibility.
        "backface-visible" => vec![("backface-visibility", "visible".into())],
        "backface-hidden" => vec![("backface-visibility", "hidden".into())],
        // forced-color-adjust.
        "forced-color-adjust-auto" => vec![("forced-color-adjust", "auto".into())],
        "forced-color-adjust-none" => vec![("forced-color-adjust", "none".into())],
        // color-scheme.
        "scheme-normal" => vec![("color-scheme", "normal".into())],
        "scheme-dark" => vec![("color-scheme", "dark".into())],
        "scheme-light" => vec![("color-scheme", "light".into())],
        "scheme-light-dark" => vec![("color-scheme", "light dark".into())],
        "scheme-only-dark" => vec![("color-scheme", "only dark".into())],
        "scheme-only-light" => vec![("color-scheme", "only light".into())],
        _ => return None,
    };
    Some(decls)
}

/// How an alignment family maps the `start`/`end` keywords.
#[derive(Clone, Copy)]
enum PosStyle {
    /// `start`→`flex-start`, `end`→`flex-end` (the flex content-/self-position
    /// group: align-content/items/self, justify-content, justify-self).
    Flex,
    /// `start`→`start`, `end`→`end` (the grid group: justify-items, place-*).
    Grid,
}

/// Which keyword values a particular alignment utility accepts, on top of the
/// always-allowed `center`/`center-safe`/`start`/`end`/`end-safe`/`stretch`.
struct AlignSpec {
    prop: &'static str,
    style: PosStyle,
    normal: bool,
    auto: bool,
    baseline: bool,
    baseline_last: bool,
    /// `between`/`around`/`evenly` (the distributed-content keywords).
    distributed: bool,
}

/// Resolves an alignment value keyword to its CSS value under `spec`, or `None`
/// when the keyword is not valid for that utility (Tailwind then emits nothing).
fn resolve_align_value(value: &str, spec: &AlignSpec) -> Option<String> {
    let flex = matches!(spec.style, PosStyle::Flex);
    let css = match value {
        "center" => "center".to_string(),
        "center-safe" => "safe center".to_string(),
        "start" => if flex { "flex-start" } else { "start" }.to_string(),
        "end" => if flex { "flex-end" } else { "end" }.to_string(),
        "end-safe" => if flex { "safe flex-end" } else { "safe end" }.to_string(),
        "stretch" => "stretch".to_string(),
        "normal" if spec.normal => "normal".to_string(),
        "auto" if spec.auto => "auto".to_string(),
        "baseline" if spec.baseline => "baseline".to_string(),
        "baseline-last" if spec.baseline_last => "last baseline".to_string(),
        "between" if spec.distributed => "space-between".to_string(),
        "around" if spec.distributed => "space-around".to_string(),
        "evenly" if spec.distributed => "space-evenly".to_string(),
        _ => return None,
    };
    Some(css)
}

/// The alignment family: `justify-content`/`justify-items`/`justify-self`,
/// `align-content`/`align-items`/`align-self`, `place-content`/`place-items`/
/// `place-self`. None of these accept a negative prefix or a `/modifier`.
///
/// Returns `None` when `base` is not one of these (fall through to the generic
/// generator — importantly `content-[…]`/`content-none`), `Some(Err(Invalid))`
/// for a token Tailwind itself rejects, and `Some(Ok(..))` for a match.
fn alignment_utility(base: &str) -> Option<Result<Utility, Fail>> {
    let negative = base.starts_with('-');
    let stem = base.strip_prefix('-').unwrap_or(base);
    let (core, has_mod) = match stem.split_once('/') {
        Some((c, _)) => (c, true),
        None => (stem, false),
    };

    // Bare family names generate nothing. (`justify-items`/`justify-self` are
    // caught below by the `justify-` fallthrough returning an unknown value.)
    if matches!(core, "place-content" | "place-items" | "place-self") {
        return Some(Err(Fail::Invalid));
    }

    // Longest family prefix wins.
    let (spec, value) = if let Some(v) = core.strip_prefix("justify-items-") {
        (
            AlignSpec {
                prop: "justify-items",
                style: PosStyle::Grid,
                normal: true,
                auto: false,
                baseline: false,
                baseline_last: false,
                distributed: false,
            },
            v,
        )
    } else if let Some(v) = core.strip_prefix("justify-self-") {
        (
            AlignSpec {
                prop: "justify-self",
                style: PosStyle::Flex,
                normal: false,
                auto: true,
                baseline: false,
                baseline_last: false,
                distributed: false,
            },
            v,
        )
    } else if let Some(v) = core.strip_prefix("justify-") {
        (
            AlignSpec {
                prop: "justify-content",
                style: PosStyle::Flex,
                normal: true,
                auto: false,
                baseline: false,
                baseline_last: false,
                distributed: true,
            },
            v,
        )
    } else if let Some(v) = core.strip_prefix("place-content-") {
        (
            AlignSpec {
                prop: "place-content",
                style: PosStyle::Grid,
                normal: false,
                auto: false,
                baseline: true,
                baseline_last: false,
                distributed: true,
            },
            v,
        )
    } else if let Some(v) = core.strip_prefix("place-items-") {
        (
            AlignSpec {
                prop: "place-items",
                style: PosStyle::Grid,
                normal: false,
                auto: false,
                baseline: true,
                baseline_last: false,
                distributed: false,
            },
            v,
        )
    } else if let Some(v) = core.strip_prefix("place-self-") {
        (
            AlignSpec {
                prop: "place-self",
                style: PosStyle::Grid,
                normal: false,
                auto: true,
                baseline: false,
                baseline_last: false,
                distributed: false,
            },
            v,
        )
    } else if let Some(v) = core.strip_prefix("content-") {
        // `content-` is shared with the CSS `content` property utility
        // (`content-['…']`, `content-none`). Only recognized align-content
        // keywords are ours; hand anything else to the generic generator.
        let spec = AlignSpec {
            prop: "align-content",
            style: PosStyle::Flex,
            normal: true,
            auto: false,
            baseline: true,
            baseline_last: false,
            distributed: true,
        };
        return Some(match resolve_align_value(v, &spec) {
            Some(css) if !negative && !has_mod => Ok(Utility::simple(vec![(spec.prop, css)])),
            Some(_) => Err(Fail::Invalid),
            None => {
                if v.starts_with('[') || v.starts_with('(') || v == "none" {
                    return None;
                }
                Err(Fail::Invalid)
            }
        });
    } else if let Some(v) = core.strip_prefix("items-") {
        (
            AlignSpec {
                prop: "align-items",
                style: PosStyle::Flex,
                normal: false,
                auto: false,
                baseline: true,
                baseline_last: true,
                distributed: false,
            },
            v,
        )
    } else if let Some(v) = core.strip_prefix("self-") {
        (
            AlignSpec {
                prop: "align-self",
                style: PosStyle::Flex,
                normal: false,
                auto: true,
                baseline: true,
                baseline_last: true,
                distributed: false,
            },
            v,
        )
    } else {
        return None;
    };

    if negative || has_mod {
        return Some(Err(Fail::Invalid));
    }
    Some(match resolve_align_value(value, &spec) {
        Some(css) => Ok(Utility::simple(vec![(spec.prop, css)])),
        None => Err(Fail::Invalid),
    })
}

/// The `mask-*` utility family. `pb` is the base with any leading `-` stripped;
/// `negative` records whether that `-` was present (only the linear/conic angle
/// utilities use it). Returns `Err(Fail::Invalid)` for every token Tailwind itself
/// rejects (a bad value, a stray `/modifier`, a negative where none is allowed),
/// so those generate nothing — matching the reference.
fn mask_utility(pb: &str, negative: bool) -> Result<Utility, Fail> {
    // No `mask-*` utility here takes an opacity/`/…` modifier; a slash makes the
    // token invalid (`mask-none/foo`, `mask-clip-border/foo`, `mask-conic-45/foo`).
    if pb.contains('/') {
        return Err(Fail::Invalid);
    }

    // The shared trailing declarations every gradient/edge mask emits.
    fn tail(decls: &mut Vec<(String, String)>) {
        decls.push(("mask-composite".into(), "intersect".into()));
        decls.push((
            "mask-image".into(),
            "var(--tw-mask-linear), var(--tw-mask-radial), var(--tw-mask-conic)".into(),
        ));
    }
    let ok = |decls: Vec<(String, String)>| {
        Ok(Utility {
            selector: SelectorKind::Class,
            decls,
            rank: 100,
        })
    };
    let d = |p: &str, v: &str| (p.to_string(), v.to_string());

    // Static keyword utilities. All reject a leading `-`.
    if !negative {
        let kw: Option<(&str, &str)> = match pb {
            "mask-none" => Some(("mask-image", "none")),
            "mask-clip-border" => Some(("mask-clip", "border-box")),
            "mask-clip-content" => Some(("mask-clip", "content-box")),
            "mask-clip-fill" => Some(("mask-clip", "fill-box")),
            "mask-clip-padding" => Some(("mask-clip", "padding-box")),
            "mask-clip-stroke" => Some(("mask-clip", "stroke-box")),
            "mask-clip-view" => Some(("mask-clip", "view-box")),
            "mask-no-clip" => Some(("mask-clip", "no-clip")),
            "mask-origin-border" => Some(("mask-origin", "border-box")),
            "mask-origin-content" => Some(("mask-origin", "content-box")),
            "mask-origin-fill" => Some(("mask-origin", "fill-box")),
            "mask-origin-padding" => Some(("mask-origin", "padding-box")),
            "mask-origin-stroke" => Some(("mask-origin", "stroke-box")),
            "mask-origin-view" => Some(("mask-origin", "view-box")),
            "mask-circle" => Some(("--tw-mask-radial-shape", "circle")),
            "mask-ellipse" => Some(("--tw-mask-radial-shape", "ellipse")),
            "mask-radial-closest-corner" => Some(("--tw-mask-radial-size", "closest-corner")),
            "mask-radial-closest-side" => Some(("--tw-mask-radial-size", "closest-side")),
            "mask-radial-farthest-corner" => Some(("--tw-mask-radial-size", "farthest-corner")),
            "mask-radial-farthest-side" => Some(("--tw-mask-radial-size", "farthest-side")),
            "mask-radial-at-top" => Some(("--tw-mask-radial-position", "top")),
            "mask-radial-at-bottom" => Some(("--tw-mask-radial-position", "bottom")),
            "mask-radial-at-left" => Some(("--tw-mask-radial-position", "left")),
            "mask-radial-at-right" => Some(("--tw-mask-radial-position", "right")),
            "mask-radial-at-top-left" => Some(("--tw-mask-radial-position", "top left")),
            "mask-radial-at-top-right" => Some(("--tw-mask-radial-position", "top right")),
            "mask-radial-at-bottom-left" => Some(("--tw-mask-radial-position", "bottom left")),
            "mask-radial-at-bottom-right" => Some(("--tw-mask-radial-position", "bottom right")),
            _ => None,
        };
        if let Some((prop, value)) = kw {
            return ok(vec![d(prop, value)]);
        }
    }

    // Linear / conic angle utilities: `mask-linear-<n>`, `mask-conic-<n>`, and
    // their negative forms `-mask-linear-<n>`.
    for (prefix, axis) in [("mask-linear-", "linear"), ("mask-conic-", "conic")] {
        // Guard the from/to gradient prefixes so they are not consumed here.
        if pb.starts_with(&format!("{prefix}from-")) || pb.starts_with(&format!("{prefix}to-")) {
            continue;
        }
        if let Some(value) = pb.strip_prefix(prefix) {
            let angle = mask_angle_value(value, negative).ok_or(Fail::Invalid)?;
            let gradient = if axis == "linear" {
                "linear-gradient(var(--tw-mask-linear-stops, var(--tw-mask-linear-position)))"
            } else {
                "conic-gradient(var(--tw-mask-conic-stops, var(--tw-mask-conic-position)))"
            };
            let mut decls = vec![
                d(&format!("--tw-mask-{axis}-position"), &angle),
                d(&format!("--tw-mask-{axis}"), gradient),
            ];
            tail(&mut decls);
            return ok(decls);
        }
    }

    // Everything below is a from/to stop and requires a non-negative position.
    if negative {
        return Err(Fail::Invalid);
    }

    // Linear / radial / conic gradient stops: `mask-<axis>-from-<v>` / `-to-<v>`.
    for axis in ["linear", "radial", "conic"] {
        for (kind, kind_prefix) in [("from", "from-"), ("to", "to-")] {
            let prefix = format!("mask-{axis}-{kind_prefix}");
            let Some(value) = pb.strip_prefix(&prefix) else {
                continue;
            };
            let pos = mask_stop_position(value).ok_or(Fail::Invalid)?;
            let stops = match axis {
                "linear" => "var(--tw-mask-linear-position), var(--tw-mask-linear-from-color) \
                     var(--tw-mask-linear-from-position), var(--tw-mask-linear-to-color) \
                     var(--tw-mask-linear-to-position)"
                    .to_string(),
                "radial" => "var(--tw-mask-radial-shape) var(--tw-mask-radial-size) at \
                     var(--tw-mask-radial-position), var(--tw-mask-radial-from-color) \
                     var(--tw-mask-radial-from-position), var(--tw-mask-radial-to-color) \
                     var(--tw-mask-radial-to-position)"
                    .to_string(),
                _ => "from var(--tw-mask-conic-position), var(--tw-mask-conic-from-color) \
                     var(--tw-mask-conic-from-position), var(--tw-mask-conic-to-color) \
                     var(--tw-mask-conic-to-position)"
                    .to_string(),
            };
            let gradient_fn = match axis {
                "linear" => "linear-gradient",
                "radial" => "radial-gradient",
                _ => "conic-gradient",
            };
            let mut decls = vec![
                d(&format!("--tw-mask-{axis}-stops"), &stops),
                d(
                    &format!("--tw-mask-{axis}"),
                    &format!("{gradient_fn}(var(--tw-mask-{axis}-stops))"),
                ),
                d(&format!("--tw-mask-{axis}-{kind}-position"), &pos),
            ];
            tail(&mut decls);
            return ok(decls);
        }
    }

    // Edge masks: `mask-t/r/b/l/x/y-from/to-<v>`. `x` = left+right, `y` = bottom+top.
    for (tok, sides) in [
        ("t", &["top"][..]),
        ("r", &["right"][..]),
        ("b", &["bottom"][..]),
        ("l", &["left"][..]),
        ("x", &["left", "right"][..]),
        ("y", &["bottom", "top"][..]),
    ] {
        let Some(rest) = pb.strip_prefix(&format!("mask-{tok}-")) else {
            continue;
        };
        let (kind, value) = if let Some(v) = rest.strip_prefix("from-") {
            ("from", v)
        } else if let Some(v) = rest.strip_prefix("to-") {
            ("to", v)
        } else {
            return Err(Fail::Invalid);
        };
        let pos = mask_stop_position(value).ok_or(Fail::Invalid)?;
        let mut decls = Vec::new();
        for side in sides {
            decls.push(d(&format!("--tw-mask-{side}-{kind}-position"), &pos));
            decls.push(d(
                &format!("--tw-mask-{side}"),
                &format!(
                    "linear-gradient(to {side}, var(--tw-mask-{side}-from-color) \
                     var(--tw-mask-{side}-from-position), var(--tw-mask-{side}-to-color) \
                     var(--tw-mask-{side}-to-position))"
                ),
            ));
        }
        decls.push(d(
            "--tw-mask-linear",
            "var(--tw-mask-left), var(--tw-mask-right), var(--tw-mask-bottom), var(--tw-mask-top)",
        ));
        tail(&mut decls);
        return ok(decls);
    }

    Err(Fail::Invalid)
}

/// A `mask-*-from/to-<v>` stop position. A bare number must be a spacing multiplier
/// (a non-negative multiple of `0.25`, canonical form) and compiles to
/// `calc(var(--spacing) * <n>)`, with `0` special-cased to `0px`. A percentage must
/// be a non-negative integer and passes through literally. Everything else (`.5`,
/// `2.8175`, `2.5%`, negatives, `unknown`) is rejected, matching Tailwind.
fn mask_stop_position(value: &str) -> Option<String> {
    if let Some(pct) = value.strip_suffix('%') {
        let ok = !pct.is_empty() && pct.bytes().all(|b| b.is_ascii_digit());
        return ok.then(|| value.to_string());
    }
    if value == "0" {
        return Some("0px".to_string());
    }
    is_spacing_multiplier(value).then(|| format!("calc(var(--spacing) * {value})"))
}

/// A non-negative multiple of `0.25` in canonical form: an integer, or an integer
/// part followed by `.25`, `.5`, or `.75`. (`1.0`, `.5`, `3.7`, `1.125` all fail.)
fn is_spacing_multiplier(s: &str) -> bool {
    match s.split_once('.') {
        Some((int, frac)) => {
            !int.is_empty()
                && int.bytes().all(|b| b.is_ascii_digit())
                && matches!(frac, "25" | "5" | "75")
        }
        None => !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()),
    }
}

/// Splits a leading Tailwind data-type hint (`length:`, `number:`, `color:`, …)
/// off an arbitrary value's inner text. The hint is lowercase letters/`-` before
/// the first `:`; `var(--x)` (no top-level `:`) and `min(1px:…)`-style functions
/// (a `(` inside the "hint") yield `None`.
fn split_data_type_hint(s: &str) -> Option<(&str, &str)> {
    let colon = s.find(':')?;
    let hint = &s[..colon];
    if !hint.is_empty() && hint.bytes().all(|b| b.is_ascii_lowercase() || b == b'-') {
        Some((hint, &s[colon + 1..]))
    } else {
        None
    }
}

/// Whether a raw value reads as a CSS `<number>` or `<percentage>`: optional sign,
/// digits with at most one decimal point, optional trailing `%`. (`0.5`, `.5`,
/// `50%`, `2` pass; `var(--x)`, `5px`, `1.2.3` fail.)
fn is_number_or_percentage(s: &str) -> bool {
    let body = s.strip_suffix('%').unwrap_or(s);
    let body = body.strip_prefix(['-', '+']).unwrap_or(body);
    if body.is_empty() {
        return false;
    }
    let mut dots = 0;
    for b in body.bytes() {
        match b {
            b'0'..=b'9' => {}
            b'.' => dots += 1,
            _ => return false,
        }
    }
    dots <= 1
}

/// A `mask-linear-<n>` / `mask-conic-<n>` angle. `<n>` must be a non-negative
/// integer. `0` -> `0deg`, `±1` -> `±1deg`, otherwise `calc(1deg * <±n>)`.
fn mask_angle_value(value: &str, negative: bool) -> Option<String> {
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if value == "0" {
        return Some("0deg".to_string());
    }
    if value == "1" {
        return Some(if negative { "-1deg" } else { "1deg" }.to_string());
    }
    let signed = if negative {
        format!("-{value}")
    } else {
        value.to_string()
    };
    Some(format!("calc(1deg * {signed})"))
}

fn unknown(full: &str) -> Fail {
    Fail::Unsupported(format!(
        "unsupported Tailwind utility class `{full}`: the native compiler does not yet generate it. Extend src/tailwind.rs (do not silently drop it)."
    ))
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
    if let Some((numerator, denominator)) = parse_fraction(value) {
        let inner = format!("calc({numerator} / {denominator} * 100%)");
        return Some(if negative {
            format!("calc({inner} * -1)")
        } else {
            inner
        });
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

/// Formats a float without a trailing `.0` and with at most four decimals.
fn format_number(value: f64) -> String {
    if (value - value.round()).abs() < 1e-9 {
        format!("{}", value.round() as i64)
    } else {
        let s = format!("{value:.4}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// The bracketed arbitrary value of a token, with Tailwind's underscore-to-space
/// rewriting (`[auto_1fr]` -> `auto 1fr`; `\_` stays a literal underscore) and
/// math-operator spacing inside math functions (`[min(800px,100dvh-280px)]` ->
/// `min(800px,100dvh - 280px)` — without the spaces the CSS is invalid).
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
    Some(space_math_operators(&out))
}

/// Inserts spaces around `+`/`-`/`*`/`/` inside CSS math functions, matching
/// Tailwind's arbitrary-value decoding (`calc(100dvw-32px)` is invalid CSS;
/// `calc(100dvw - 32px)` is what the reference emits). Non-math function
/// arguments (`var(--x)`) are left untouched, as is a sign that starts a value
/// (`calc(-1px + 2em)`).
fn space_math_operators(value: &str) -> String {
    const MATH_FNS: &[&str] = &[
        "calc", "min", "max", "clamp", "mod", "rem", "round", "pow", "sqrt", "hypot", "log", "exp",
        "abs", "sign", "atan2",
    ];
    let bytes = value.as_bytes();
    let mut out = String::with_capacity(value.len());
    // Whether each open paren belongs to a math function.
    let mut stack: Vec<bool> = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        match b {
            b'(' => {
                let name_end = out.len();
                let name_start = out
                    .rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '-'))
                    .map(|p| p + 1)
                    .unwrap_or(0);
                let name = out[name_start..name_end].to_ascii_lowercase();
                stack.push(MATH_FNS.contains(&name.as_str()));
                out.push('(');
            }
            b')' => {
                stack.pop();
                out.push(')');
            }
            b'+' | b'-' | b'*' | b'/' if stack.last().copied().unwrap_or(false) => {
                let trimmed = out.trim_end();
                let prev = trimmed.chars().next_back();
                let operand_before = matches!(
                    prev,
                    Some(c) if c.is_ascii_alphanumeric() || c == '%' || c == ')'
                );
                // `1e-5` / `1E+5`: an exponent sign, not an operator.
                let exponent = matches!(b, b'+' | b'-')
                    && matches!(prev, Some('e' | 'E'))
                    && trimmed
                        .chars()
                        .rev()
                        .nth(1)
                        .is_some_and(|c| c.is_ascii_digit());
                if operand_before && !exponent {
                    while out.ends_with(' ') {
                        out.pop();
                    }
                    out.push(' ');
                    out.push(b as char);
                    out.push(' ');
                    while i + 1 < bytes.len() && bytes[i + 1] == b' ' {
                        i += 1;
                    }
                } else {
                    out.push(b as char);
                }
            }
            other => out.push(other as char),
        }
        i += 1;
    }
    out
}

/// Sizing utilities (`w-`, `h-`, `min-*`, `max-*`, `size-`). Returns `Ok(None)`
/// when the prefix is not a sizing family.
fn sizing_utility(base: &str, _full: &str, theme: &Theme) -> Result<Option<Utility>, Fail> {
    // The final flag marks CSS logical-property families (`min-inline-size`, …),
    // which resolve values with slightly different axis rules than the physical
    // `w`/`h` families (no `screen-<bp>` scale, container scale on the inline
    // axis only, no `--max-width-*` namespace).
    let families: [(&str, &[&str], char, SizeKind, bool); 13] = [
        ("w", &["width"], 'w', SizeKind::Plain, false),
        ("h", &["height"], 'h', SizeKind::Plain, false),
        ("min-w", &["min-width"], 'w', SizeKind::Min, false),
        ("min-h", &["min-height"], 'h', SizeKind::Min, false),
        ("max-w", &["max-width"], 'w', SizeKind::Max, false),
        ("max-h", &["max-height"], 'h', SizeKind::Max, false),
        ("size", &["width", "height"], 's', SizeKind::Plain, false),
        ("min-inline", &["min-inline-size"], 'w', SizeKind::Min, true),
        ("max-inline", &["max-inline-size"], 'w', SizeKind::Max, true),
        ("min-block", &["min-block-size"], 'h', SizeKind::Min, true),
        ("max-block", &["max-block-size"], 'h', SizeKind::Max, true),
        // Logical block/inline sizes. Bare `block`/`inline` (and `inline-block`,
        // `inline-flex`, …) are display keywords resolved earlier by
        // `keyword_utility`; only `block-<value>`/`inline-<value>` reach here.
        ("block", &["block-size"], 'h', SizeKind::Plain, true),
        ("inline", &["inline-size"], 'w', SizeKind::Plain, true),
    ];
    for (prefix, properties, axis, kind, logical) in families {
        let Some(value) = strip_family(base, prefix) else {
            continue;
        };
        // A value Tailwind cannot resolve (e.g. `w-none`, `h--1`, `w-1/-2`) makes
        // it generate nothing at all; mirror that with `Fail::Invalid`.
        let resolved = if logical {
            logical_size_value(value, axis, kind, theme)
        } else {
            size_value(value, axis, kind, theme)
        }
        .ok_or(Fail::Invalid)?;
        let decls = properties
            .iter()
            .map(|p| (*p, resolved.clone()))
            .collect::<Vec<_>>();
        return Ok(Some(Utility::simple(decls)));
    }
    Ok(None)
}

/// Whether a sizing family is a `min-*`, `max-*`, or the plain `w`/`h`/`size`.
/// Governs which keyword values are accepted (`auto` on all but `max-*`, `none`
/// only on `max-*`).
#[derive(Clone, Copy, PartialEq)]
enum SizeKind {
    Plain,
    Min,
    Max,
}

/// A sizing value: spacing steps, `px`, `full`, `screen` (axis-aware), `auto`,
/// `none`, `fit`/`min`/`max` content keywords, `lh` (height only), fractions
/// (kept as `calc(<n> / <d> * 100%)`), the container scale
/// (`sm` -> `var(--container-sm)`), `screen-<bp>` (`var(--breakpoint-md)`), and
/// arbitrary lengths.
fn size_value(value: &str, axis: char, kind: SizeKind, theme: &Theme) -> Option<String> {
    match value {
        "px" => return Some("1px".to_string()),
        "full" => return Some("100%".to_string()),
        // `auto` is valid everywhere except the `max-*` families.
        "auto" => return (kind != SizeKind::Max).then(|| "auto".to_string()),
        // `none` is only valid on the `max-*` families (`max-w-none`).
        "none" => return (kind == SizeKind::Max).then(|| "none".to_string()),
        "fit" => return Some("fit-content".to_string()),
        "max" => return Some("max-content".to_string()),
        "min" => return Some("min-content".to_string()),
        // `lh` (one line-height unit) is a height-axis-only keyword.
        "lh" => return (axis == 'h').then(|| "1lh".to_string()),
        "screen" => {
            return Some(match axis {
                'h' => "100vh".to_string(),
                'w' => "100vw".to_string(),
                _ => return None,
            });
        }
        // Dynamic/small/large viewport units (`min-h-dvh`, `w-dvw`, …) are
        // valid on every sizing axis in v4.
        "dvh" | "dvw" | "svh" | "svw" | "lvh" | "lvw" => {
            return Some(format!("100{value}"));
        }
        _ => {}
    }
    if let Some(bp) = value.strip_prefix("screen-") {
        let var = format!("--breakpoint-{bp}");
        return theme.contains(&var).then(|| format!("var({var})"));
    }
    // The `--max-width-*` namespace (e.g. `max-w-prose` -> `65ch`) is
    // max-width-only and inlines its literal theme value.
    if kind == SizeKind::Max
        && axis == 'w'
        && let Some(v) = theme.get(&format!("--max-width-{value}"))
    {
        return Some(v.to_string());
    }
    let container = format!("--container-{value}");
    if theme.contains(&container) {
        return Some(format!("var({container})"));
    }
    // Fractions stay as an unfolded `calc(<n> / <d> * 100%)` (with spaces),
    // matching Tailwind v4's emitted form.
    if let Some((n, d)) = parse_fraction(value) {
        return Some(format!("calc({n} / {d} * 100%)"));
    }
    if let Some(inner) = arbitrary_value(value) {
        return Some(inner);
    }
    spacing_value(value, false)
}

/// Value resolution for the CSS logical-property sizing families
/// (`min-inline-size`, `max-inline-size`, `min-block-size`, `max-block-size`).
/// Shares the keyword/spacing/fraction/arbitrary handling of [`size_value`] but
/// with the logical-axis rules Tailwind v4 uses: `screen` resolves by axis
/// (`100vh` on block, `100vw` on inline) with no `screen-<bp>` breakpoint scale;
/// the container scale (`xl` -> `var(--container-xl)`) applies on the inline
/// (width) axis only; `lh` is block-axis only; and there is no `--max-width-*`
/// namespace.
fn logical_size_value(value: &str, axis: char, kind: SizeKind, theme: &Theme) -> Option<String> {
    match value {
        "px" => return Some("1px".to_string()),
        "full" => return Some("100%".to_string()),
        // `auto` is valid on every family except `max-*`.
        "auto" => return (kind != SizeKind::Max).then(|| "auto".to_string()),
        // `none` is only valid on the `max-*` families.
        "none" => return (kind == SizeKind::Max).then(|| "none".to_string()),
        "fit" => return Some("fit-content".to_string()),
        "max" => return Some("max-content".to_string()),
        "min" => return Some("min-content".to_string()),
        // `lh` (one line-height unit) is a block-axis-only keyword.
        "lh" => return (axis == 'h').then(|| "1lh".to_string()),
        "screen" => {
            return Some(match axis {
                'h' => "100vh".to_string(),
                'w' => "100vw".to_string(),
                _ => return None,
            });
        }
        // Viewport units are axis-restricted on logical properties: the block
        // (height) axis takes only the `*vh` units, the inline (width) axis only
        // the `*vw` units.
        "dvh" | "svh" | "lvh" => return (axis == 'h').then(|| format!("100{value}")),
        "dvw" | "svw" | "lvw" => return (axis == 'w').then(|| format!("100{value}")),
        _ => {}
    }
    // The container scale is inline-axis (width-like) only; the block axis rejects
    // it (`max-block-xl` generates nothing).
    if axis == 'w' {
        let container = format!("--container-{value}");
        if theme.contains(&container) {
            return Some(format!("var({container})"));
        }
    }
    // Fractions stay as an unfolded `calc(<n> / <d> * 100%)` (with spaces).
    if let Some((n, d)) = parse_fraction(value) {
        return Some(format!("calc({n} / {d} * 100%)"));
    }
    if let Some(inner) = arbitrary_value(value) {
        return Some(inner);
    }
    spacing_value(value, false)
}

/// `grid-template-*` values: `none`, `subgrid`, a positive track count
/// (`12` -> `repeat(12, minmax(0, 1fr))`), or an arbitrary track list. A zero or
/// non-positive count is rejected (Tailwind generates nothing for `grid-cols-0`).
fn grid_template_value(value: &str) -> Option<String> {
    match value {
        "none" => return Some("none".to_string()),
        "subgrid" => return Some("subgrid".to_string()),
        _ => {}
    }
    if is_bare_integer(value) {
        // `grid-cols-0` is invalid — the count must be at least 1.
        if value.bytes().any(|b| b != b'0') {
            return Some(format!("repeat({value}, minmax(0, 1fr))"));
        }
        return None;
    }
    arbitrary_value(value)
}

/// Whether a token is a bare, unsigned base-10 integer (no sign, no decimal).
fn is_bare_integer(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit())
}

/// The `flex` / `basis` / `grow` / `shrink` / `order` / `grid` / `col` / `row`
/// utility families. `base` is the class with any leading `-` already stripped;
/// `negative` records whether that `-` was present (only `order`, `col`, and
/// `row` accept it). Returns `Ok(None)` when `base` belongs to none of these
/// families (fall through), `Err(Fail::Invalid)` for a candidate Tailwind itself
/// rejects (generating nothing), and `Ok(Some(_))` for a real utility.
fn flex_grid_utility(
    base: &str,
    _full: &str,
    negative: bool,
    theme: &Theme,
) -> Result<Option<Utility>, Fail> {
    // Bare `flex`/`grid` are display keywords (handled positively in
    // `keyword_utility`); they reach here only as the non-negatable `-flex` /
    // `-grid`, which are invalid.
    if base == "flex" || base == "grid" {
        return Err(Fail::Invalid);
    }
    // --- flex-<n> / flex-<fraction> / flex-auto|none|initial (keywords handled
    //     in `keyword_utility`; this covers the dynamic + arbitrary forms). ---
    if let Some(v) = base.strip_prefix("flex-") {
        if negative {
            return Err(Fail::Invalid);
        }
        let value = match v {
            "auto" => "auto".to_string(),
            "initial" => "0 auto".to_string(),
            "none" => "none".to_string(),
            _ => {
                if let Some((n, d)) = parse_fraction(v) {
                    format!("calc({n}/{d} * 100%)")
                } else if is_bare_integer(v) {
                    v.to_string()
                } else if let Some(inner) = arbitrary_value(v) {
                    inner
                } else {
                    return Err(Fail::Invalid);
                }
            }
        };
        return Ok(Some(Utility::simple(vec![("flex", value)])));
    }

    // --- basis-<value>: flex-basis. Fractions are kept as an unfolded
    //     `calc(<n> / <d> * 100%)` (with spaces, unlike the `flex` shorthand). ---
    if base == "basis" {
        return Err(Fail::Invalid);
    }
    if let Some(v) = base.strip_prefix("basis-") {
        if negative {
            return Err(Fail::Invalid);
        }
        let value = match v {
            "auto" => "auto".to_string(),
            "full" => "100%".to_string(),
            "px" => "1px".to_string(),
            _ => {
                if let Some((n, d)) = parse_fraction(v) {
                    format!("calc({n} / {d} * 100%)")
                } else if theme.contains(&format!("--container-{v}")) {
                    format!("var(--container-{v})")
                } else if let Some(sp) = spacing_value(v, false) {
                    sp
                } else if let Some(inner) = arbitrary_value(v) {
                    inner
                } else {
                    return Err(Fail::Invalid);
                }
            }
        };
        return Ok(Some(Utility::simple(vec![("flex-basis", value)])));
    }

    // --- grow / shrink: flex-grow / flex-shrink (non-negatable, integer only). ---
    for (name, property) in [("grow", "flex-grow"), ("shrink", "flex-shrink")] {
        if base == name {
            if negative {
                return Err(Fail::Invalid);
            }
            return Ok(Some(Utility::simple(vec![(property, "1".to_string())])));
        }
        if let Some(v) = base.strip_prefix(name).and_then(|r| r.strip_prefix('-')) {
            if negative {
                return Err(Fail::Invalid);
            }
            let value = if is_bare_integer(v) {
                v.to_string()
            } else if let Some(inner) = arbitrary_value(v) {
                inner
            } else {
                return Err(Fail::Invalid);
            };
            return Ok(Some(Utility::simple(vec![(property, value)])));
        }
    }

    // --- order-<n> / order-first|last|none (negatable: `-order-4`). ---
    if base == "order" {
        return Err(Fail::Invalid);
    }
    if let Some(v) = base.strip_prefix("order-") {
        let value = match v {
            "first" if !negative => "-9999".to_string(),
            "last" if !negative => "9999".to_string(),
            "none" if !negative => "0".to_string(),
            _ => {
                if is_bare_integer(v) {
                    negate_line(v, negative)
                } else if let Some(inner) = arbitrary_value(v) {
                    negate_line(&inner, negative)
                } else {
                    return Err(Fail::Invalid);
                }
            }
        };
        return Ok(Some(Utility::simple(vec![("order", value)])));
    }

    // --- grid-cols / grid-rows / grid-flow / auto-cols / auto-rows. ---
    if base == "grid-cols" || base == "grid-rows" {
        return Err(Fail::Invalid);
    }
    if let Some(v) = base.strip_prefix("grid-cols-") {
        if negative {
            return Err(Fail::Invalid);
        }
        let value = grid_template_value(v).ok_or(Fail::Invalid)?;
        return Ok(Some(Utility::simple(vec![(
            "grid-template-columns",
            value,
        )])));
    }
    if let Some(v) = base.strip_prefix("grid-rows-") {
        if negative {
            return Err(Fail::Invalid);
        }
        let value = grid_template_value(v).ok_or(Fail::Invalid)?;
        return Ok(Some(Utility::simple(vec![("grid-template-rows", value)])));
    }
    if base == "grid-flow" {
        return Err(Fail::Invalid);
    }
    if let Some(v) = base.strip_prefix("grid-flow-") {
        if negative {
            return Err(Fail::Invalid);
        }
        let value = match v {
            "row" => "row",
            "col" => "column",
            "dense" => "dense",
            "row-dense" => "row dense",
            "col-dense" => "column dense",
            _ => return Err(Fail::Invalid),
        };
        return Ok(Some(Utility::simple(vec![(
            "grid-auto-flow",
            value.to_string(),
        )])));
    }
    for (prefix, property) in [
        ("auto-cols-", "grid-auto-columns"),
        ("auto-rows-", "grid-auto-rows"),
    ] {
        if let Some(v) = base.strip_prefix(prefix) {
            if negative {
                return Err(Fail::Invalid);
            }
            let value = match v {
                "auto" => "auto".to_string(),
                "min" => "min-content".to_string(),
                "max" => "max-content".to_string(),
                "fr" => "minmax(0, 1fr)".to_string(),
                _ => arbitrary_value(v).ok_or(Fail::Invalid)?,
            };
            return Ok(Some(Utility::simple(vec![(property, value)])));
        }
    }

    // --- col-* -> grid-column(-start|-end); row-* -> grid-row(-start|-end).
    //     Span / start / end are matched before the bare axis prefix. ---
    for (axis, main, start, end) in [
        ("col", "grid-column", "grid-column-start", "grid-column-end"),
        ("row", "grid-row", "grid-row-start", "grid-row-end"),
    ] {
        let span_bare = format!("{axis}-span");
        let span_prefix = format!("{axis}-span-");
        if base == span_bare {
            return Err(Fail::Invalid);
        }
        if let Some(v) = base.strip_prefix(span_prefix.as_str()) {
            if negative {
                return Err(Fail::Invalid);
            }
            let value = if v == "full" {
                "1 / -1".to_string()
            } else if is_bare_integer(v) {
                format!("span {v} / span {v}")
            } else if let Some(inner) = arbitrary_value(v) {
                format!("span {inner} / span {inner}")
            } else {
                return Err(Fail::Invalid);
            };
            return Ok(Some(Utility::simple(vec![(main, value)])));
        }
        for (kind, property) in [("start", start), ("end", end)] {
            let bare = format!("{axis}-{kind}");
            let prefix = format!("{axis}-{kind}-");
            if base == bare {
                return Err(Fail::Invalid);
            }
            if let Some(v) = base.strip_prefix(prefix.as_str()) {
                let value = grid_line_value(v, negative).ok_or(Fail::Invalid)?;
                return Ok(Some(Utility::simple(vec![(property, value)])));
            }
        }
        // Bare `col`/`row` is invalid; `col-<n>` / `col-auto` set grid-column.
        if base == axis {
            return Err(Fail::Invalid);
        }
        if let Some(v) = base.strip_prefix(format!("{axis}-").as_str()) {
            let value = grid_line_value(v, negative).ok_or(Fail::Invalid)?;
            return Ok(Some(Utility::simple(vec![(main, value)])));
        }
    }

    Ok(None)
}

/// A `grid-column`/`grid-row` line value: `auto` (non-negatable), a bare integer,
/// or an arbitrary value — with an optional leading `-` folded to `calc(<v> * -1)`.
fn grid_line_value(value: &str, negative: bool) -> Option<String> {
    if value == "auto" {
        return (!negative).then(|| "auto".to_string());
    }
    if is_bare_integer(value) {
        return Some(negate_line(value, negative));
    }
    arbitrary_value(value).map(|inner| negate_line(&inner, negative))
}

/// Applies a Tailwind negative to a line/order value: `calc(<v> * -1)`.
fn negate_line(value: &str, negative: bool) -> String {
    if negative {
        format!("calc({value} * -1)")
    } else {
        value.to_string()
    }
}

/// Border utilities: bare side shorthands (`border`, `border-b`), side widths
/// (`border-l-2`), plain widths (`border-2`), side colors (`border-l-amber-500`),
/// and colors (`border-transparent`, `border-[color:var(--x)]`).
fn border_utility(
    base: &str,
    full: &str,
    theme: &Theme,
    tw_props: &mut BTreeSet<TwProp>,
) -> Result<Utility, Fail> {
    let rest = base.strip_prefix("border").unwrap_or("");
    let rest = rest.strip_prefix('-').unwrap_or(rest);

    // A remaining leading `-` (e.g. `border--0`) is a negative width, which
    // Tailwind rejects outright.
    if rest.starts_with('-') {
        return Err(Fail::Invalid);
    }

    // `border-spacing` / `border-spacing-x` / `border-spacing-y` (table cell
    // spacing) — resolved from the spacing scale.
    if rest == "spacing" || rest.starts_with("spacing-") {
        return border_spacing_utility(rest, full, tw_props);
    }

    // Bare `border` / `border-<side>`.
    if let Some(props) = border_side_decls(rest, "1px") {
        tw_props.insert(TwProp::BorderStyle);
        return Ok(Utility::ranked(
            props,
            if rest.is_empty() { 40 } else { 41 },
        ));
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
    if let Some(value) = color_value(rest, theme) {
        return Ok(Utility::ranked(vec![("border-color", value)], 42));
    }
    // Arbitrary widths/colors are an engine gap; any other unrecognized token
    // (a stray `/modifier`, an unknown value) is one Tailwind rejects outright.
    if rest.starts_with('[') {
        return Err(unknown(full));
    }
    Err(Fail::Invalid)
}

/// `border-spacing[-x|-y]-<n>`: sets `--tw-border-spacing-x`/`y` from the spacing
/// scale (both @property registered) plus the `border-spacing` shorthand.
fn border_spacing_utility(
    rest: &str,
    _full: &str,
    tw_props: &mut BTreeSet<TwProp>,
) -> Result<Utility, Fail> {
    // Which axis (or both), and the spacing step.
    let (set_x, set_y, step) = if let Some(step) = rest.strip_prefix("spacing-x-") {
        (true, false, step)
    } else if let Some(step) = rest.strip_prefix("spacing-y-") {
        (false, true, step)
    } else if let Some(step) = rest.strip_prefix("spacing-") {
        (true, true, step)
    } else {
        // Bare `border-spacing` / `border-spacing-x` — Tailwind generates nothing.
        return Err(Fail::Invalid);
    };
    let value = spacing_value(step, false).ok_or(Fail::Invalid)?;
    // Both custom properties are always registered.
    tw_props.insert(TwProp::BorderSpacingX);
    tw_props.insert(TwProp::BorderSpacingY);
    let mut decls: Vec<(String, String)> = Vec::new();
    if set_x {
        decls.push(("--tw-border-spacing-x".to_string(), value.clone()));
    }
    if set_y {
        decls.push(("--tw-border-spacing-y".to_string(), value.clone()));
    }
    decls.push((
        "border-spacing".to_string(),
        "var(--tw-border-spacing-x) var(--tw-border-spacing-y)".to_string(),
    ));
    Ok(Utility {
        selector: SelectorKind::Class,
        decls,
        rank: 100,
    })
}

/// The declarations a `from-*`/`via-*`/`to-*` color stop emits under the **v3**
/// dialect, whose gradient composition is structurally different from v4's:
///
/// * every stop carries its own position (`<color> var(--tw-gradient-…-position)`)
///   instead of the position living in a separate `--tw-gradient-position`;
/// * `from-*` and `via-*` additionally reset `--tw-gradient-to` to the same color
///   at zero alpha, so a two-stop gradient fades out rather than falling back to
///   v4's `#0000`;
/// * `via-*` inlines its color into `--tw-gradient-stops` — v3 has no
///   `--tw-gradient-via` or `--tw-gradient-via-stops`;
/// * `to-*` sets only `--tw-gradient-to`, never `--tw-gradient-stops`.
///
/// KNOWN GAP (shared with every other v3 colour utility, not specific to
/// gradients): a `/<pct>` opacity modifier still resolves through [`color_value`],
/// which compiles it to v4's `color-mix(in oklab, …)`. v3 writes
/// `rgb(<r> <g> <b> / <alpha>)`, and browsers serialize the two differently in
/// `getComputedStyle`. Closing it means threading the dialect through
/// `color_value` and its ~30 call sites, which is a separate change; no app in
/// the corpus exercises it. See FINDINGS item 24.
fn v3_gradient_stop_decls(
    family: &str,
    token: &str,
    theme: &Theme,
) -> Option<Vec<(&'static str, String)>> {
    let resolved = color_value(token, theme)?;
    let (color_token, _) = split_color_modifier(token);
    let faded = v3_transparent_color(color_srgb_literal(color_token, theme).as_deref());
    Some(match family {
        "from" => vec![
            (
                "--tw-gradient-from",
                format!("{resolved} var(--tw-gradient-from-position)"),
            ),
            (
                "--tw-gradient-to",
                format!("{faded} var(--tw-gradient-to-position)"),
            ),
            (
                "--tw-gradient-stops",
                "var(--tw-gradient-from), var(--tw-gradient-to)".to_string(),
            ),
        ],
        "via" => vec![
            (
                "--tw-gradient-to",
                format!("{faded} var(--tw-gradient-to-position)"),
            ),
            (
                "--tw-gradient-stops",
                format!(
                    "var(--tw-gradient-from), {resolved} var(--tw-gradient-via-position), var(--tw-gradient-to)"
                ),
            ),
        ],
        _ => vec![(
            "--tw-gradient-to",
            format!("{resolved} var(--tw-gradient-to-position)"),
        )],
    })
}

/// v3's `transparentTo(color)`: the same color at zero alpha, spelled
/// `rgb(<r> <g> <b> / 0)`. A color v3's own parser cannot read (`currentColor`,
/// `inherit`, a bare `var(…)`) falls back to `rgb(255 255 255 / 0)`, exactly as
/// upstream does.
fn v3_transparent_color(literal: Option<&str>) -> String {
    const FALLBACK: &str = "rgb(255 255 255 / 0)";
    let Some(literal) = literal else {
        return FALLBACK.to_string();
    };
    let literal = literal.trim();
    if literal.eq_ignore_ascii_case("transparent") {
        // v3 parses `transparent` as `rgba(0,0,0,0)`.
        return "rgb(0 0 0 / 0)".to_string();
    }
    if let Some(hex) = literal.strip_prefix('#')
        && let Some((r, g, b)) = parse_hex_rgb(hex)
    {
        return format!("rgb({r} {g} {b} / 0)");
    }
    // `rgb(…)` / `rgba(…)`: keep the three channels, drop the alpha.
    if let Some(rest) = literal
        .strip_prefix("rgb(")
        .or_else(|| literal.strip_prefix("rgba("))
        && let Some(inner) = rest.strip_suffix(')')
    {
        let channels: Vec<&str> = inner
            .split([',', ' ', '/'])
            .filter(|part| !part.is_empty())
            .collect();
        if channels.len() >= 3 {
            return format!("rgb({} {} {} / 0)", channels[0], channels[1], channels[2]);
        }
    }
    FALLBACK.to_string()
}

/// The 8-bit sRGB channels of a `#rgb` / `#rgba` / `#rrggbb` / `#rrggbbaa` body
/// (the `#` already stripped). `None` for any other shape.
fn parse_hex_rgb(hex: &str) -> Option<(u8, u8, u8)> {
    if !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let pair = |s: &str| u8::from_str_radix(s, 16).ok();
    match hex.len() {
        3 | 4 => {
            let d: Vec<char> = hex.chars().collect();
            Some((
                pair(&format!("{}{}", d[0], d[0]))?,
                pair(&format!("{}{}", d[1], d[1]))?,
                pair(&format!("{}{}", d[2], d[2]))?,
            ))
        }
        6 | 8 => Some((pair(&hex[0..2])?, pair(&hex[2..4])?, pair(&hex[4..6])?)),
        _ => None,
    }
}

/// The `--tw-gradient-{from,via,to}` declaration(s) for a color stop token (with
/// optional `/<pct>` modifier). A plain token yields one declaration; a modifier
/// yields the `color-mix(in oklab, …)` value plus the static sRGB fallback
/// Tailwind emits for browsers without `oklab`.
fn gradient_color_decls(
    prop: &'static str,
    token: &str,
    theme: &Theme,
) -> Option<Vec<(&'static str, String)>> {
    let resolved = color_value(token, theme)?;
    let (color_token, modifier) = split_color_modifier(token);
    match modifier {
        None => Some(vec![(prop, resolved)]),
        Some(pct) => {
            // The static fallback resolves the theme color to its literal value.
            let literal = match color_token {
                "transparent" => "transparent".to_string(),
                "current" => "currentcolor".to_string(),
                "inherit" => "inherit".to_string(),
                _ if color_token.starts_with('[') => arbitrary_value(color_token)?,
                _ => theme.get(&format!("--color-{color_token}"))?.to_string(),
            };
            Some(vec![
                (prop, resolved),
                (
                    prop,
                    format!("color-mix(in srgb, {literal} {pct}%, transparent)"),
                ),
            ])
        }
    }
}

/// The `--tw-gradient-*-position` property a `from-`/`via-`/`to-` stop
/// position sets.
fn gradient_position_property(family: &str) -> &'static str {
    match family {
        "from" => "--tw-gradient-from-position",
        "via" => "--tw-gradient-via-position",
        _ => "--tw-gradient-to-position",
    }
}

fn px_width(n: &str) -> String {
    if n == "0" {
        "0".to_string()
    } else {
        format!("{n}px")
    }
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

/// A border-radius scale value: the bare default, theme sizes, `full`, `none`,
/// and arbitrary lengths.
fn radius_value(size: &str, theme: &Theme) -> Option<String> {
    // The theme wins over v4's built-in literals. v4 hard-codes `rounded`/`rounded-full`/
    // `rounded-none` because its own theme carries no token for them — but a legacy v3
    // config resolves them as real tokens with different values (`rounded-full` is
    // `9999px` in v3, `calc(infinity * 1px)` in v4), and taking the literal first made
    // every v3 avatar a 33554432px-radius circle instead of a 9999px one.
    // (`rounded` with no size stays inlined: v4's own `--radius` lives in its
    // `@theme default inline reference` block, whose tokens the reference build
    // substitutes rather than emitting as `var()`.)
    if !size.is_empty() {
        let var = format!("--radius-{size}");
        if theme.contains(&var) {
            return Some(format!("var({var})"));
        }
    }
    match size {
        "" => return Some("0.25rem".to_string()),
        "full" => return Some("calc(infinity * 1px)".to_string()),
        "none" => return Some("0".to_string()),
        _ => {}
    }
    arbitrary_value(size)
}

/// Output rank for a rounded side/corner keyword, ordering overlapping
/// families the way Tailwind does (whole box < logical sides < physical sides
/// < logical corners < physical corners).
fn rounded_side_rank(side: &str) -> Option<u16> {
    Some(match side {
        "s" => 46,
        "e" => 47,
        "t" => 48,
        "r" => 49,
        "b" => 50,
        "l" => 51,
        "ss" => 52,
        "se" => 53,
        "ee" => 54,
        "es" => 55,
        "tl" => 56,
        "tr" => 57,
        "br" => 58,
        "bl" => 59,
        _ => return None,
    })
}

/// The border-radius properties a rounded side/corner keyword sets, in
/// Tailwind's emission order.
fn rounded_side_properties(side: &str) -> &'static [&'static str] {
    match side {
        "s" => &["border-start-start-radius", "border-end-start-radius"],
        "e" => &["border-start-end-radius", "border-end-end-radius"],
        "t" => &["border-top-left-radius", "border-top-right-radius"],
        "r" => &["border-top-right-radius", "border-bottom-right-radius"],
        "b" => &["border-bottom-right-radius", "border-bottom-left-radius"],
        "l" => &["border-top-left-radius", "border-bottom-left-radius"],
        "ss" => &["border-start-start-radius"],
        "se" => &["border-start-end-radius"],
        "ee" => &["border-end-end-radius"],
        "es" => &["border-end-start-radius"],
        "tl" => &["border-top-left-radius"],
        "tr" => &["border-top-right-radius"],
        "br" => &["border-bottom-right-radius"],
        "bl" => &["border-bottom-left-radius"],
        _ => &[],
    }
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
    dialect: Dialect,
) -> Result<Utility, Fail> {
    let rest = base.strip_prefix("ring").unwrap_or("");
    let rest = rest.strip_prefix('-').unwrap_or(rest);
    if rest.is_empty() || (rest.bytes().all(|b| b.is_ascii_digit()) && !rest.is_empty()) {
        let width = if rest.is_empty() { "1" } else { rest };
        register_shadow_group(tw_props);
        return Ok(Utility::simple(vec![
            (
                "--tw-ring-shadow",
                format!(
                    "var(--tw-ring-inset,) 0 0 0 calc({width}px + var(--tw-ring-offset-width)) var(--tw-ring-color, currentcolor)"
                ),
            ),
            ("box-shadow", dialect.box_shadow_chain().to_string()),
        ]));
    }
    if rest == "inset" {
        register_shadow_group(tw_props);
        return Ok(Utility::simple(vec![(
            "--tw-ring-inset",
            "inset".to_string(),
        )]));
    }
    if rest == "offset" || rest.starts_with("offset-") {
        let n = rest.strip_prefix("offset-").unwrap_or("");
        // ring-offset-<width>: a plain integer.
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
        // ring-offset-<color> (with optional `/<pct>` modifier).
        if let Some(decls) = color_prop_decls("--tw-ring-offset-color", n, theme) {
            register_shadow_group(tw_props);
            return Ok(Utility::simple(decls));
        }
        // Arbitrary widths/colors are an engine gap; everything else (bare
        // `ring-offset`, `ring-offset-inset`, `ring-offset-unknown`) is a token
        // Tailwind rejects outright.
        if n.starts_with('[') {
            return Err(unknown(full));
        }
        return Err(Fail::Invalid);
    }
    if let Some(decls) = color_prop_decls("--tw-ring-color", rest, theme) {
        register_shadow_group(tw_props);
        return Ok(Utility::simple(decls));
    }
    if rest.starts_with('[') {
        return Err(unknown(full));
    }
    Err(Fail::Invalid)
}

/// A color utility's declaration(s) for `prop`: a plain token yields one
/// `(prop, value)`; a `/<pct>` modifier yields the `color-mix(in oklab, …)` value
/// plus the static `color-mix(in srgb, …)` fallback Tailwind emits for browsers
/// without `oklab`. Returns `None` when the token is not a color.
fn color_prop_decls(
    prop: &'static str,
    token: &str,
    theme: &Theme,
) -> Option<Vec<(&'static str, String)>> {
    let resolved = color_value(token, theme)?;
    let (color_token, modifier) = split_color_modifier(token);
    match modifier {
        None => Some(vec![(prop, resolved)]),
        Some(pct) => {
            let literal = color_srgb_literal(color_token, theme)?;
            Some(vec![
                (prop, resolved),
                (
                    prop,
                    format!("color-mix(in srgb, {literal} {pct}%, transparent)"),
                ),
            ])
        }
    }
}

/// The literal sRGB-fallback color for a color token (resolving a theme color to
/// its published value), used by the static `color-mix(in srgb, …)` fallback.
fn color_srgb_literal(color_token: &str, theme: &Theme) -> Option<String> {
    Some(match color_token {
        "transparent" => "transparent".to_string(),
        "current" => "currentcolor".to_string(),
        "inherit" => "inherit".to_string(),
        _ if color_token.starts_with('[') => arbitrary_value(color_token)?,
        _ => theme.get(&format!("--color-{color_token}"))?.to_string(),
    })
}

/// Divide utilities: `divide-x`/`divide-y` (per-axis child border widths with a
/// `reverse` flag), `divide-<style>`, and `divide-<color>`. All target the
/// between-children selector `:where(.divide-* > :not(:last-child))`.
fn divide_utility(
    base: &str,
    full: &str,
    theme: &Theme,
    tw_props: &mut BTreeSet<TwProp>,
    dialect: Dialect,
) -> Result<Utility, Fail> {
    let rest = match base.strip_prefix("divide-") {
        Some(rest) => rest,
        // Bare `divide` is not a utility; Tailwind generates nothing.
        None => return Err(Fail::Invalid),
    };

    // `divide-x` / `divide-y` axis widths and the `reverse` flag.
    for (axis, reverse_var, reverse_prop) in [
        ('x', "--tw-divide-x-reverse", TwProp::DivideXReverse),
        ('y', "--tw-divide-y-reverse", TwProp::DivideYReverse),
    ] {
        let prefix = format!("{axis}");
        if rest == prefix
            || rest
                .strip_prefix(&prefix)
                .is_some_and(|r| r.starts_with('-'))
        {
            let suffix = rest.strip_prefix(&prefix).unwrap_or("");
            let suffix = suffix.strip_prefix('-').unwrap_or(suffix);
            if suffix == "reverse" {
                tw_props.insert(reverse_prop);
                return Ok(Utility {
                    selector: SelectorKind::SpaceChildren,
                    decls: vec![(reverse_var.to_string(), "1".to_string())],
                    rank: 100,
                });
            }
            // `divide-x` (empty suffix) is width 1px; `divide-x-<n>` is `<n>px`
            // (including `divide-x-0` -> `0px`). Only bare integers are valid.
            let width = if suffix.is_empty() {
                "1".to_string()
            } else if !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()) {
                suffix.to_string()
            } else {
                return Err(Fail::Invalid);
            };
            tw_props.insert(reverse_prop);
            // v3 selects every child but the FIRST, so its widths sit on the
            // opposite edges from v4's (same inversion as `space-*` above), it
            // uses physical sides, and it leans on the preflight's
            // `border-style: solid` instead of `--tw-border-style`.
            if dialect == Dialect::V3 {
                let (reversed_side, normal_side) = if axis == 'x' {
                    ("border-right-width", "border-left-width")
                } else {
                    ("border-bottom-width", "border-top-width")
                };
                let width = if width == "0" {
                    "0px".to_string()
                } else {
                    format!("{width}px")
                };
                let mut decls = vec![(reverse_var.to_string(), "0".to_string())];
                // v3 writes the leading edge first on the y axis, the trailing one
                // first on x — the order upstream's plugin emits.
                let ordered: [(&str, bool); 2] = if axis == 'x' {
                    [(reversed_side, true), (normal_side, false)]
                } else {
                    [(normal_side, false), (reversed_side, true)]
                };
                for (prop, carries_reverse) in ordered {
                    let val = if carries_reverse {
                        format!("calc({width} * var({reverse_var}))")
                    } else {
                        format!("calc({width} * calc(1 - var({reverse_var})))")
                    };
                    decls.push((prop.to_string(), val));
                }
                return Ok(Utility {
                    selector: SelectorKind::SpaceChildren,
                    decls,
                    rank: 100,
                });
            }
            tw_props.insert(TwProp::BorderStyle);
            let decls = if axis == 'x' {
                vec![
                    (reverse_var.to_string(), "0".to_string()),
                    (
                        "border-inline-style".to_string(),
                        "var(--tw-border-style)".to_string(),
                    ),
                    (
                        "border-inline-start-width".to_string(),
                        format!("calc({width}px * var({reverse_var}))"),
                    ),
                    (
                        "border-inline-end-width".to_string(),
                        format!("calc({width}px * calc(1 - var({reverse_var})))"),
                    ),
                ]
            } else {
                vec![
                    (reverse_var.to_string(), "0".to_string()),
                    (
                        "border-bottom-style".to_string(),
                        "var(--tw-border-style)".to_string(),
                    ),
                    (
                        "border-top-style".to_string(),
                        "var(--tw-border-style)".to_string(),
                    ),
                    (
                        "border-top-width".to_string(),
                        format!("calc({width}px * var({reverse_var}))"),
                    ),
                    (
                        "border-bottom-width".to_string(),
                        format!("calc({width}px * calc(1 - var({reverse_var})))"),
                    ),
                ]
            };
            return Ok(Utility {
                selector: SelectorKind::SpaceChildren,
                decls,
                rank: 100,
            });
        }
    }

    // `divide-<style>`: writes `--tw-border-style` and `border-style` (v3 has no
    // `--tw-border-style` and writes only the real property).
    if let Some(style) = divide_border_style(rest) {
        let mut decls = Vec::new();
        if dialect == Dialect::V4 {
            decls.push(("--tw-border-style".to_string(), style.to_string()));
        }
        decls.push(("border-style".to_string(), style.to_string()));
        return Ok(Utility {
            selector: SelectorKind::SpaceChildren,
            decls,
            rank: 100,
        });
    }

    // `divide-<color>`.
    if let Some(decls) = color_prop_decls("border-color", rest, theme) {
        return Ok(Utility {
            selector: SelectorKind::SpaceChildren,
            decls: decls.into_iter().map(|(p, v)| (p.to_string(), v)).collect(),
            rank: 100,
        });
    }
    if rest.starts_with('[') {
        return Err(unknown(full));
    }
    Err(Fail::Invalid)
}

/// The border-style keyword a `divide-<style>` sets, or `None`.
fn divide_border_style(rest: &str) -> Option<&'static str> {
    Some(match rest {
        "solid" => "solid",
        "dashed" => "dashed",
        "dotted" => "dotted",
        "double" => "double",
        "none" => "none",
        _ => return None,
    })
}

/// Outline utilities: `outline` (width 1px), `outline-<width>`, `outline-none`,
/// `outline-hidden`, `outline-<style>`, `outline-offset-<n>` (the only negatable
/// member), and `outline-<color>`.
fn outline_utility(
    base: &str,
    full: &str,
    negative: bool,
    theme: &Theme,
    tw_props: &mut BTreeSet<TwProp>,
) -> Result<Utility, Fail> {
    // `outline-offset-<value>` — the one outline utility that accepts a `-`.
    if base == "outline-offset" || base.starts_with("outline-offset-") {
        let Some(v) = base.strip_prefix("outline-offset-") else {
            return Err(Fail::Invalid);
        };
        // A `/modifier` is invalid on outline-offset.
        if v.contains('/') {
            return Err(Fail::Invalid);
        }
        if !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()) {
            let value = if negative {
                format!("calc({v}px * -1)")
            } else {
                format!("{v}px")
            };
            return Ok(Utility::simple(vec![("outline-offset", value)]));
        }
        if v.starts_with('[') {
            if let Some(inner) = arbitrary_value(v) {
                let value = if negative {
                    format!("calc({inner} * -1)")
                } else {
                    inner
                };
                return Ok(Utility::simple(vec![("outline-offset", value)]));
            }
            return Err(unknown(full));
        }
        return Err(Fail::Invalid);
    }

    // Every other outline utility rejects a leading `-`.
    if negative {
        return Err(Fail::Invalid);
    }

    if base == "outline" {
        tw_props.insert(TwProp::OutlineStyle);
        return Ok(Utility::simple(vec![
            ("outline-style", "var(--tw-outline-style)".to_string()),
            ("outline-width", "1px".to_string()),
        ]));
    }
    if base == "outline-none" {
        return Ok(Utility::simple(vec![
            ("--tw-outline-style", "none".to_string()),
            ("outline-style", "none".to_string()),
        ]));
    }
    if base == "outline-hidden" {
        return Ok(Utility::simple(vec![
            ("--tw-outline-style", "none".to_string()),
            ("outline-style", "none".to_string()),
            ("outline", "2px solid transparent".to_string()),
            ("outline-offset", "2px".to_string()),
        ]));
    }
    let rest = base.strip_prefix("outline-").unwrap_or("");

    // `outline-<style>`.
    if let Some(style) = outline_style_keyword(rest) {
        return Ok(Utility::simple(vec![
            ("--tw-outline-style", style.to_string()),
            ("outline-style", style.to_string()),
        ]));
    }
    // `outline-<width>`: a bare integer.
    if !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()) {
        tw_props.insert(TwProp::OutlineStyle);
        return Ok(Utility::simple(vec![
            ("outline-style", "var(--tw-outline-style)".to_string()),
            ("outline-width", format!("{rest}px")),
        ]));
    }
    // `outline-<color>` (with optional `/<pct>` modifier).
    if let Some(decls) = color_prop_decls("outline-color", rest, theme) {
        return Ok(Utility::simple(decls));
    }
    // Arbitrary outline width/color are an engine gap.
    if rest.starts_with('[') {
        return Err(unknown(full));
    }
    Err(Fail::Invalid)
}

/// The outline-style keyword a `outline-<style>` sets, or `None`.
fn outline_style_keyword(rest: &str) -> Option<&'static str> {
    Some(match rest {
        "solid" => "solid",
        "dashed" => "dashed",
        "dotted" => "dotted",
        "double" => "double",
        _ => return None,
    })
}

/// Rewrites each color inside a shadow value into
/// `var(--tw-shadow-color, <color>)` (with a space after the comma, as Tailwind's
/// box-shadow utilities emit), matching the compiled shadow utilities.
fn wrap_shadow_colors(value: &str) -> String {
    wrap_colors(value, "--tw-shadow-color", true)
}

/// Rewrites each color inside an inset-shadow value into
/// `var(--tw-inset-shadow-color, <color>)` (space after the comma).
fn wrap_inset_shadow_colors(value: &str) -> String {
    wrap_colors(value, "--tw-inset-shadow-color", true)
}

/// A shadow-family color utility's declarations (`shadow-<color>`,
/// `inset-shadow-<color>`): the color composed with the utility's alpha var via
/// `color-mix(in oklab, …)`, plus the static sRGB-fallback line Tailwind emits for
/// browsers without `oklab`. `None` when the token is not a color.
fn shadow_color_decls(
    prop: &'static str,
    alpha_var: &str,
    token: &str,
    theme: &Theme,
) -> Option<Vec<(&'static str, String)>> {
    let resolved = color_value(token, theme)?;
    let (color_token, modifier) = split_color_modifier(token);
    let literal = color_srgb_literal(color_token, theme)?;
    let first = format!("color-mix(in oklab, {resolved} var({alpha_var}), transparent)");
    let second = match modifier {
        None => literal,
        Some(pct) => format!("color-mix(in srgb, {literal} {pct}%, transparent)"),
    };
    Some(vec![(prop, first), (prop, second)])
}

/// Rewrites each color inside a text-shadow value into
/// `var(--tw-text-shadow-color, <color>)` (with a space after the comma, as
/// Tailwind's text-shadow utilities emit).
fn wrap_text_shadow_colors(value: &str) -> String {
    wrap_colors(value, "--tw-text-shadow-color", true)
}

/// Rewrites every color token in a shadow-like value list (hex literals and
/// color-function calls) into `var(<var>, <color>)`. Lengths, `inset`, and
/// non-color functions (`var(…)`, `calc(…)`) pass through untouched. `spaced`
/// inserts a space after the `var()` fallback comma.
fn wrap_colors(value: &str, var: &str, spaced: bool) -> String {
    let sep = if spaced { ", " } else { "," }; // the `var()` fallback separator
    const COLOR_FNS: &[&str] = &[
        "rgb", "rgba", "hsl", "hsla", "hwb", "lab", "lch", "oklab", "oklch", "color",
    ];
    let bytes = value.as_bytes();
    let mut out = String::with_capacity(value.len());
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'#' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j].is_ascii_hexdigit() {
                j += 1;
            }
            out.push_str(&format!("var({var}{sep}{})", &value[i..j]));
            i = j;
        } else if b.is_ascii_alphabetic() {
            let mut j = i;
            while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'-') {
                j += 1;
            }
            let name = &value[i..j];
            if j < bytes.len() && bytes[j] == b'(' {
                let mut depth = 0i32;
                let mut k = j;
                while k < bytes.len() {
                    match bytes[k] {
                        b'(' => depth += 1,
                        b')' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    k += 1;
                }
                let end = (k + 1).min(bytes.len());
                let call = &value[i..end];
                if COLOR_FNS.contains(&name) {
                    out.push_str(&format!("var({var}{sep}{call})"));
                } else {
                    out.push_str(call);
                }
                i = end;
            } else {
                out.push_str(name);
                i = j;
            }
        } else {
            out.push(b as char);
            i += 1;
        }
    }
    out
}

/// A theme drop-shadow value as chained `drop-shadow(…)` calls, one per
/// comma-separated layer. `sized` wraps each layer's color for
/// `--tw-drop-shadow-color`.
fn drop_shadow_layers(value: &str, sized: bool) -> String {
    split_top_level_commas(value)
        .into_iter()
        .map(|layer| {
            let layer = layer.trim();
            if sized {
                format!(
                    "drop-shadow({})",
                    wrap_colors(layer, "--tw-drop-shadow-color", false)
                )
            } else {
                format!("drop-shadow({layer})")
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Splits a value on commas outside parentheses and brackets.
fn split_top_level_commas(value: &str) -> Vec<&str> {
    let bytes = value.as_bytes();
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            b',' if depth == 0 => {
                parts.push(&value[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&value[start..]);
    parts
}

/// The transition-property composites, verbatim from Tailwind v4.
fn transition_utility(base: &str) -> Option<Vec<(&'static str, String)>> {
    let property = match base {
        "transition" => {
            "color, background-color, border-color, outline-color, text-decoration-color, fill, stroke, --tw-gradient-from, --tw-gradient-via, --tw-gradient-to, opacity, box-shadow, transform, translate, scale, rotate, filter, -webkit-backdrop-filter, backdrop-filter, display, content-visibility, overlay, pointer-events"
        }
        "transition-colors" => {
            "color, background-color, border-color, outline-color, text-decoration-color, fill, stroke, --tw-gradient-from, --tw-gradient-via, --tw-gradient-to"
        }
        "transition-opacity" => "opacity",
        "transition-transform" => "transform, translate, scale, rotate",
        "transition-shadow" => "box-shadow",
        "transition-all" => "all",
        "transition-none" => {
            return Some(vec![("transition-property", "none".to_string())]);
        }
        "transition-normal" => {
            return Some(vec![("transition-behavior", "normal".to_string())]);
        }
        "transition-discrete" => {
            return Some(vec![("transition-behavior", "allow-discrete".to_string())]);
        }
        _ => return None,
    };
    Some(vec![
        ("transition-property", property.to_string()),
        (
            "transition-timing-function",
            "var(--tw-ease, var(--default-transition-timing-function))".to_string(),
        ),
        (
            "transition-duration",
            "var(--tw-duration, var(--default-transition-duration))".to_string(),
        ),
    ])
}

/// Padding/margin/gap utilities over the spacing scale. Margins additionally
/// accept `auto` and negatives. Returns `Ok(None)` when the prefix is not a
/// spacing family, `Err` when it is but the step is invalid.
fn spacing_utility(base: &str, full: &str, negative: bool) -> Result<Option<Utility>, Fail> {
    let families: [(&str, &str, bool, u16); 23] = [
        ("gap-", "gap", false, 100),
        ("p-", "padding", false, 20),
        ("px-", "padding-inline", false, 21),
        ("py-", "padding-block", false, 22),
        ("pt-", "padding-top", false, 23),
        ("pr-", "padding-right", false, 24),
        ("pb-", "padding-bottom", false, 25),
        ("pl-", "padding-left", false, 26),
        ("ps-", "padding-inline-start", false, 27),
        ("pe-", "padding-inline-end", false, 28),
        ("pbs-", "padding-block-start", false, 29),
        ("pbe-", "padding-block-end", false, 30),
        ("m-", "margin", true, 10),
        ("mx-", "margin-inline", true, 11),
        ("my-", "margin-block", true, 12),
        ("mt-", "margin-top", true, 13),
        ("mr-", "margin-right", true, 14),
        ("mb-", "margin-bottom", true, 15),
        ("ml-", "margin-left", true, 16),
        ("ms-", "margin-inline-start", true, 17),
        ("me-", "margin-inline-end", true, 18),
        ("mbs-", "margin-block-start", true, 19),
        ("mbe-", "margin-block-end", true, 110),
    ];
    for (prefix, property, is_margin, rank) in families {
        if let Some(step) = base.strip_prefix(prefix) {
            // Padding/gap have no negative form: Tailwind rejects a leading `-`
            // (generates nothing, not a hard error).
            if negative && !is_margin {
                return Err(Fail::Invalid);
            }
            if is_margin && step == "auto" {
                if negative {
                    return Err(unknown(full));
                }
                return Ok(Some(Utility::ranked(
                    vec![(property, "auto".to_string())],
                    rank,
                )));
            }
            if step == "px" {
                let value = if negative { "-1px" } else { "1px" };
                return Ok(Some(Utility::ranked(
                    vec![(property, value.to_string())],
                    rank,
                )));
            }
            if let Some(inner) = arbitrary_value(step) {
                let value = if negative {
                    format!("calc({inner} * -1)")
                } else {
                    inner
                };
                return Ok(Some(Utility::ranked(vec![(property, value)], rank)));
            }
            // A non-canonical step (`px-big`, `px-.75`, `px-0.375`, `px-2.50`)
            // is a value Tailwind rejects outright -> generates nothing.
            let value = spacing_value(step, negative).ok_or(Fail::Invalid)?;
            return Ok(Some(Utility::ranked(vec![(property, value)], rank)));
        }
    }
    Ok(None)
}

/// The spacing-scale value for a numeric step (integers or halves like `1.5`),
/// matching Tailwind's compiled output: `0` -> `0`, `1` -> `var(--spacing)`,
/// otherwise `calc(var(--spacing) * n)`.
fn spacing_value(step: &str, negative: bool) -> Option<String> {
    // The step must be a non-negative multiple of `0.25` in canonical form
    // (`2`, `1.5`, `0.25`, `2.75`); over-precise or trailing-zero decimals
    // (`0.375`, `2.50`) and non-numeric tokens (`big`, `.75`) are rejected,
    // matching Tailwind.
    if !is_spacing_multiplier(step) {
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
    } else if token.starts_with('[') {
        // A bare arbitrary value: for a COLOR utility this is `bg-[#383838]`,
        // `bg-[rgb(0_0_0)]`, `border-[hsl(...)]`. But `color_value` is shared with
        // `text-`, where `text-[11px]` is a SIZE — so only accept a value that clearly
        // reads as a color (a hex or a CSS color function); anything else returns
        // `None` and falls through to the size/length family. `var(...)` without the
        // explicit `[color:var(...)]` hint is likewise left ambiguous (unhandled here).
        let inner = arbitrary_value(token)?;
        if !looks_like_color(&inner) {
            return None;
        }
        inner
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
/// Whether a bare arbitrary value reads as a color literal (a hex, or a CSS color
/// function) rather than a length/number — used to keep `text-[11px]` (a size) out of
/// the color path while letting `bg-[#383838]` / `text-[rgb(...)]` through.
fn looks_like_color(value: &str) -> bool {
    let v = value.trim();
    v.starts_with('#')
        || [
            "rgb(", "rgba(", "hsl(", "hsla(", "oklch(", "oklab(", "lab(", "lch(", "hwb(", "color(",
        ]
        .iter()
        .any(|func| v.starts_with(func))
}

fn split_color_modifier(token: &str) -> (&str, Option<String>) {
    if let Some(pos) = token.rfind('/') {
        // The `/` must be OUTSIDE any arbitrary-value brackets (a `bg-[url(a/b)]`
        // slash is part of the value, not an opacity modifier).
        let after_brackets = token[..pos].matches('[').count() == token[..pos].matches(']').count();
        if !after_brackets {
            return (token, None);
        }
        let suffix = &token[pos + 1..];
        if suffix.is_empty() {
            return (token, None);
        }
        // Arbitrary bracketed opacity: `/[.04]`, `/[0.145]`, `/[50%]`.
        if let Some(inner) = suffix.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            return match arbitrary_opacity_percent(inner) {
                Some(pct) => (&token[..pos], Some(pct)),
                None => (token, None),
            };
        }
        // Bare numeric percentage: `/10` -> "10".
        if suffix.bytes().all(|b| b.is_ascii_digit() || b == b'.') {
            return (&token[..pos], Some(suffix.to_string()));
        }
    }
    (token, None)
}

/// The percentage NUMBER (no `%`) for a bracketed color-opacity modifier's inner
/// text: `50%` -> `50`, a bare fraction `.04` -> `4` (×100, as Tailwind does),
/// `0.145` -> `14.5`. Returns `None` if it is not a plain number/percentage.
fn arbitrary_opacity_percent(inner: &str) -> Option<String> {
    let inner = inner.trim();
    if let Some(pct) = inner.strip_suffix('%') {
        return Some(pct.trim().to_string());
    }
    let fraction: f64 = inner.parse().ok()?;
    // Round to 4 decimals to shed binary-float noise (0.145×100 = 14.4999…), then
    // trim trailing zeros and a dangling dot: 14.5000 -> 14.5, 4.0000 -> 4.
    let mut s = format!("{:.4}", fraction * 100.0);
    if s.contains('.') {
        s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    }
    Some(s)
}

fn is_cursor_keyword(kw: &str) -> bool {
    matches!(
        kw,
        "auto"
            | "default"
            | "pointer"
            | "wait"
            | "text"
            | "move"
            | "help"
            | "not-allowed"
            | "none"
            | "context-menu"
            | "progress"
            | "cell"
            | "crosshair"
            | "vertical-text"
            | "alias"
            | "copy"
            | "no-drop"
            | "grab"
            | "grabbing"
            | "all-scroll"
            | "col-resize"
            | "row-resize"
            | "n-resize"
            | "e-resize"
            | "s-resize"
            | "w-resize"
            | "ne-resize"
            | "nw-resize"
            | "se-resize"
            | "sw-resize"
            | "ew-resize"
            | "ns-resize"
            | "nesw-resize"
            | "nwse-resize"
            | "zoom-in"
            | "zoom-out"
    )
}

/// Whether `base` (already stripped of any leading `-`) belongs to a
/// typography/interactivity family that has no negative form. Used by the
/// group's negative gate to reject `-cursor-*`, `-text-*`, `-font-*`, … the way
/// Tailwind does (generating nothing) rather than hard-erroring.
fn is_typography_interactivity_family(base: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "text-",
        "font-",
        "leading-",
        "tracking-",
        "list-",
        "decoration-",
        "underline-",
        "whitespace-",
        "align-",
        "accent-",
        "caret-",
        "cursor-",
        "pointer-events-",
        "select-",
        "touch-",
        "will-change-",
        "fill-",
        "stroke-",
        "line-clamp-",
        "hyphens-",
        "wrap-",
        "content-",
    ];
    PREFIXES.iter().any(|p| base.starts_with(p))
        || matches!(base, "underline" | "resize" | "appearance")
        || base.starts_with("resize-")
        || base.starts_with("appearance-")
}

/// The `font-stretch-<value>` CSS value: a named keyword, or an integer
/// percentage in the 50–200% range. Anything else is rejected.
fn font_stretch_value(core: &str) -> Option<String> {
    if matches!(
        core,
        "ultra-condensed"
            | "extra-condensed"
            | "condensed"
            | "semi-condensed"
            | "normal"
            | "semi-expanded"
            | "expanded"
            | "extra-expanded"
            | "ultra-expanded"
    ) {
        return Some(core.to_string());
    }
    let digits = core.strip_suffix('%')?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let n: u32 = digits.parse().ok()?;
    if (50..=200).contains(&n) {
        Some(format!("{n}%"))
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
        "xs" | "sm"
            | "base"
            | "lg"
            | "xl"
            | "2xl"
            | "3xl"
            | "4xl"
            | "5xl"
            | "6xl"
            | "7xl"
            | "8xl"
            | "9xl"
    )
}

/// Escapes a class name for use in a selector: every byte outside
/// `[A-Za-z0-9_-]` is backslash-escaped, and a LEADING digit (or a digit right
/// after a leading `-`) is escaped as its hex code point + a space, per CSS.escape —
/// a CSS identifier may not start with an unescaped digit, so `2xl:flex` becomes
/// `\32 xl\:flex` (matching Tailwind), not the invalid `2xl\:flex`.
fn escape_class(class: &str) -> String {
    let mut out = String::with_capacity(class.len());
    let chars: Vec<char> = class.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if c.is_ascii_digit() && (i == 0 || (i == 1 && chars[0] == '-')) {
            // ASCII '0'..='9' are code points 0x30..=0x39, so the hex is "3" + the
            // digit; the trailing space terminates the escape.
            out.push_str("\\3");
            out.push(c);
            out.push(' ');
            continue;
        }
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
    /// `@utility` definitions the app registers.
    custom_utilities: CustomUtilities,
    /// The `@layer base` body: the app's base rules (with `@apply` expanded) plus
    /// their `dark:` companions as `@media (prefers-color-scheme: dark)` rules
    /// (or the custom `dark` template's selector form when one is defined).
    base_layer: String,
    /// The `@layer components` body, expanded the same way.
    components_layer: String,
    /// The app's own `@layer utilities` body, expanded the same way. Emitted
    /// AFTER the generated utilities, which is where the app wrote it relative
    /// to its `@import 'tailwindcss'`.
    utilities_layer: String,
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
    seed_variants: std::collections::BTreeMap<String, String>,
    dialect: Dialect,
) -> Result<UserCss, String> {
    let mut base_layer = String::new();
    let mut components_layer = String::new();
    let mut utilities_layer = String::new();
    let mut postlude = String::new();
    let mut defined_classes = BTreeSet::new();
    let items = parse_top_level(css)?;
    // Variants first: a `@custom-variant` applies to the whole sheet no matter
    // where it appears. `seed_variants` are the ones the theme source carries (a
    // legacy JS config's `darkMode`); the app's own CSS overrides them.
    let mut custom_variants = seed_variants;
    // `@utility` definitions likewise apply sheet-wide, and must be registered
    // before any `@apply` in a `@layer base` rule can reference one.
    let mut custom_utilities = CustomUtilities::default();
    for item in &items {
        match item {
            TopItem::CustomVariant { name, template } => {
                custom_variants.insert(name.clone(), template.clone());
            }
            TopItem::Utility { name, body } => {
                if let Some(prefix) = name.strip_suffix("-*") {
                    if prefix.is_empty() || prefix.contains('*') {
                        return Err(format!(
                            "malformed functional utility name `@utility {name}`"
                        ));
                    }
                    custom_utilities
                        .functional
                        .insert(prefix.to_string(), body.clone());
                } else {
                    if name.contains('*') {
                        return Err(format!(
                            "`@utility {name}`: a `*` is only allowed as the `-*` suffix of a functional utility"
                        ));
                    }
                    custom_utilities.statics.insert(name.clone(), body.clone());
                }
            }
            _ => {}
        }
    }
    // The `dark:` companion of an expanded rule: under a custom `dark` variant
    // it is a sibling rule with the template applied to each selector; under
    // the default it is a `prefers-color-scheme` media block.
    let emit_dark =
        |out: &mut String,
         selector: &str,
         dark_rule: &str,
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
                out.push_str("@media (prefers-color-scheme: dark){");
                out.push_str(dark_rule);
                out.push('}');
            }
        };
    for item in items {
        match item {
            TopItem::Import => {}
            TopItem::CustomVariant { .. } => {}
            // Registered above; a definition emits nothing on its own — only the
            // candidates that match it do.
            TopItem::Utility { .. } => {}
            // `@theme` tokens are merged into the theme in `compile_with_theme` and
            // emitted (when referenced) by the theme layer — nothing to output here.
            TopItem::Theme => {}
            TopItem::Verbatim(block) => {
                collect_selector_classes(&block, &mut defined_classes);
                postlude.push_str(&block);
            }
            TopItem::Layer { names, body } => {
                // Tailwind v4's cascade layers an app may write into. Each one is
                // expanded identically and emitted into the matching output layer,
                // so a rule keeps the priority the app gave it.
                let out = match names.split_whitespace().next() {
                    Some("base") => &mut base_layer,
                    Some("components") => &mut components_layer,
                    Some("utilities") => &mut utilities_layer,
                    _ => {
                        return Err(format!(
                            "unsupported at-rule `@layer {names}` in Tailwind CSS entry (native compiler handles `@layer base`, `@layer components` and `@layer utilities`)"
                        ));
                    }
                };
                let rules = parse_rules(&body)?;
                for rule in rules {
                    collect_selector_classes(&rule.selector, &mut defined_classes);
                    let (main, dark) =
                        expand_rule(&rule, theme, tw_props, &custom_utilities, dialect)?;
                    if let Some(main) = main {
                        out.push_str(&main);
                    }
                    if let Some(dark) = dark {
                        emit_dark(out, &rule.selector, &dark, &custom_variants);
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
                let (main, dark) = expand_rule(&rule, theme, tw_props, &custom_utilities, dialect)?;
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
        custom_utilities,
        base_layer,
        components_layer,
        utilities_layer,
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

// ---------------------------------------------------------------------------
// `@utility` — app-defined utilities
// ---------------------------------------------------------------------------

/// The `@utility` definitions an app's stylesheet registers.
///
/// Tailwind v4 has two forms. A STATIC one (`@utility tab-4 { tab-size: 4 }`)
/// matches one exact class name. A FUNCTIONAL one (`@utility tab-* { … }`) owns a
/// whole family: the class carries a value after the prefix, and the body's
/// `--value(…)` calls decide which spellings of that value are accepted.
#[derive(Default)]
struct CustomUtilities {
    /// name -> body, for `@utility <name> { … }`.
    statics: BTreeMap<String, String>,
    /// prefix -> body, for `@utility <prefix>-* { … }` (the `-*` is not stored).
    functional: BTreeMap<String, String>,
}

/// A candidate matched against the registered `@utility` definitions.
struct CustomMatch<'a> {
    body: &'a str,
    /// The value segment after a functional utility's prefix (`4`, `[3px]`),
    /// or `None` for a static utility (whose body may not call `--value(…)`).
    value: Option<&'a str>,
    /// The `/…` modifier, when the candidate carries one.
    modifier: Option<&'a str>,
}

impl CustomUtilities {
    fn is_empty(&self) -> bool {
        self.statics.is_empty() && self.functional.is_empty()
    }

    /// Matches a candidate base (variants and `!` already stripped) against the
    /// registered definitions. A static name wins over a functional prefix, and
    /// among functional prefixes the LONGEST match wins (so `@utility stack-y-*`
    /// beats `@utility stack-*` for `stack-y-4`, exactly as Tailwind resolves it).
    fn lookup<'a>(&'a self, base: &'a str) -> Option<CustomMatch<'a>> {
        if let Some(body) = self.statics.get(base) {
            return Some(CustomMatch {
                body,
                value: None,
                modifier: None,
            });
        }
        let (core, modifier) = split_utility_modifier(base);
        let mut best: Option<(&String, &String)> = None;
        for (prefix, body) in &self.functional {
            let Some(rest) = core.strip_prefix(prefix.as_str()) else {
                continue;
            };
            if !rest.starts_with('-') || rest.len() < 2 {
                continue;
            }
            if best.is_none_or(|(previous, _)| prefix.len() > previous.len()) {
                best = Some((prefix, body));
            }
        }
        let (prefix, body) = best?;
        Some(CustomMatch {
            body,
            value: Some(&core[prefix.len() + 1..]),
            modifier,
        })
    }
}

/// Splits a candidate base into its value and its `/<modifier>`, ignoring a `/`
/// inside an arbitrary value (`[url(a/b)]`) or a function call.
fn split_utility_modifier(base: &str) -> (&str, Option<&str>) {
    let bytes = base.as_bytes();
    let mut depth = 0i32;
    let mut at = None;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'[' | b'(' => depth += 1,
            b']' | b')' => depth -= 1,
            b'/' if depth == 0 => at = Some(i),
            _ => {}
        }
    }
    match at {
        Some(i) => (&base[..i], Some(&base[i + 1..])),
        None => (base, None),
    }
}

/// One statement inside a `@utility` body.
enum UtilityItem {
    /// `prop: value`
    Decl { prop: String, value: String },
    /// `@apply <classes>`
    Apply(String),
    /// A nested rule or at-rule: `&:hover { … }`, `@media (…) { … }`.
    Block { prelude: String, body: String },
}

/// Splits a `@utility` body into declarations, `@apply` statements and nested
/// blocks. Anything else (a `;`-terminated at-rule) is a hard error naming it.
fn parse_utility_items(body: &str, name: &str) -> Result<Vec<UtilityItem>, String> {
    let bytes = body.as_bytes();
    let mut items = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        i = skip_ws_and_comments(body, i);
        if i >= bytes.len() {
            break;
        }
        // Find the end of this statement: a `;` or a `{` at nesting depth 0.
        let mut j = i;
        let mut depth = 0i32;
        let mut stop = None;
        while j < bytes.len() {
            match bytes[j] {
                b'(' | b'[' => depth += 1,
                b')' | b']' => depth -= 1,
                b';' if depth == 0 => {
                    stop = Some((b';', j));
                    break;
                }
                b'{' if depth == 0 => {
                    stop = Some((b'{', j));
                    break;
                }
                _ => {}
            }
            j += 1;
        }
        match stop {
            Some((b'{', brace)) => {
                let prelude = body[i..brace]
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ");
                let (inner, end) = read_braced(body, brace)?;
                items.push(UtilityItem::Block {
                    prelude,
                    body: inner,
                });
                i = end;
            }
            other => {
                let end = match other {
                    Some((_, semi)) => semi,
                    None => bytes.len(),
                };
                let statement = body[i..end].trim();
                i = if end < bytes.len() { end + 1 } else { end };
                if statement.is_empty() {
                    continue;
                }
                if let Some(classes) = statement.strip_prefix("@apply") {
                    items.push(UtilityItem::Apply(classes.trim().to_string()));
                    continue;
                }
                if statement.starts_with('@') {
                    return Err(format!(
                        "`@utility {name}`: unsupported statement {statement:?} (only declarations, `@apply`, and nested rules are)"
                    ));
                }
                let Some((prop, value)) = statement.split_once(':') else {
                    return Err(format!(
                        "`@utility {name}`: {statement:?} is neither a declaration (`prop: value`) nor a nested rule"
                    ));
                };
                items.push(UtilityItem::Decl {
                    prop: prop.trim().to_string(),
                    value: value.trim().to_string(),
                });
            }
        }
    }
    Ok(items)
}

/// A `@utility` body with its value functions resolved.
struct ExpandedUtilityBody {
    /// Top-level declarations, in source order.
    decls: Vec<(String, String)>,
    /// Nested rules/at-rules, already serialized (CSS nesting, `&`-rooted —
    /// which is exactly what Tailwind emits for these).
    nested: String,
}

impl ExpandedUtilityBody {
    fn is_empty(&self) -> bool {
        self.decls.is_empty() && self.nested.is_empty()
    }
}

/// Expands one `@utility` body against a candidate's value and modifier.
///
/// Tailwind's rule for functional utilities: a declaration whose `--value(…)` or
/// `--modifier(…)` call resolves to nothing is DROPPED, and a utility whose whole
/// body drops does not match the candidate at all. That is how the two-line form
///
/// ```css
/// @utility stack-y-* { margin-top: --spacing(--value(integer));
///                      margin-top: --value([length]); }
/// ```
///
/// accepts both `stack-y-4` and `stack-y-[3px]` while generating nothing for
/// `stack-y-px`.
#[allow(clippy::too_many_arguments)]
fn expand_utility_body(
    body: &str,
    name: &str,
    value: Option<&str>,
    modifier: Option<&str>,
    theme: &Theme,
    tw_props: &mut BTreeSet<TwProp>,
    dialect: Dialect,
    bang: &str,
) -> Result<ExpandedUtilityBody, Fail> {
    let items = parse_utility_items(body, name).map_err(Fail::Unsupported)?;
    let mut decls: Vec<(String, String)> = Vec::new();
    let mut nested = String::new();
    for item in items {
        match item {
            UtilityItem::Decl { prop, value: text } => {
                let Some(resolved) =
                    substitute_value_functions(&text, name, value, modifier, theme)?
                else {
                    continue;
                };
                decls.push((prop, resolved));
            }
            UtilityItem::Apply(classes) => {
                for class in classes.split_whitespace() {
                    let (applied, important) = split_important(class);
                    let utility = generate_utility(applied, class, theme, tw_props, dialect)?;
                    for (prop, value) in utility.decls {
                        decls.push((prop, with_important(&value, important)));
                    }
                }
            }
            UtilityItem::Block {
                prelude,
                body: inner,
            } => {
                let Some(prelude) =
                    substitute_value_functions(&prelude, name, value, modifier, theme)?
                else {
                    continue;
                };
                let expanded = expand_utility_body(
                    &inner, name, value, modifier, theme, tw_props, dialect, bang,
                )?;
                if expanded.is_empty() {
                    continue;
                }
                nested.push_str(&prelude);
                nested.push('{');
                for (prop, value) in &expanded.decls {
                    nested.push_str(&format!("{prop}:{value}{bang};"));
                }
                nested.push_str(&expanded.nested);
                nested.push('}');
            }
        }
    }
    Ok(ExpandedUtilityBody { decls, nested })
}

/// Substitutes Tailwind's CSS value functions (`--value()`, `--modifier()`,
/// `--spacing()`) in a `@utility` body fragment.
///
/// `Ok(None)` means a `--value()`/`--modifier()` call matched nothing, so the
/// enclosing declaration is dropped (Tailwind's own rule). `Err` is an engine gap
/// — a function or data type this compiler does not implement — and is never
/// silently swallowed.
fn substitute_value_functions(
    text: &str,
    name: &str,
    value: Option<&str>,
    modifier: Option<&str>,
    theme: &Theme,
) -> Result<Option<String>, Fail> {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        // A Tailwind value function is a `--ident(` at an identifier boundary;
        // `var(--spacing)` is a plain custom-property reference and is left alone.
        let boundary = i == 0
            || !(bytes[i - 1].is_ascii_alphanumeric()
                || bytes[i - 1] == b'-'
                || bytes[i - 1] == b'_');
        if !(boundary && text[i..].starts_with("--")) {
            out.push(text[i..].chars().next().expect("index is a char boundary"));
            i += text[i..].chars().next().map_or(1, char::len_utf8);
            continue;
        }
        let mut j = i + 2;
        while j < bytes.len()
            && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'-' || bytes[j] == b'_')
        {
            j += 1;
        }
        if j >= bytes.len() || bytes[j] != b'(' {
            out.push_str(&text[i..j]);
            i = j;
            continue;
        }
        let function = &text[i..j];
        let (args, end) = read_parenthesized(text, j).map_err(Fail::Unsupported)?;
        match function {
            "--value" | "--modifier" => {
                let subject = if function == "--value" {
                    value
                } else {
                    modifier
                };
                let Some(resolved) = resolve_value_call(&args, subject, name, function, theme)?
                else {
                    return Ok(None);
                };
                out.push_str(&resolved);
            }
            "--spacing" => {
                let Some(inner) = substitute_value_functions(&args, name, value, modifier, theme)?
                else {
                    return Ok(None);
                };
                if !theme.contains("--spacing") {
                    return Err(Fail::Unsupported(format!(
                        "`@utility {name}`: `--spacing({inner})` needs a `--spacing` theme variable, and this theme defines none"
                    )));
                }
                out.push_str(&format!("calc(var(--spacing) * {})", inner.trim()));
            }
            other => {
                return Err(Fail::Unsupported(format!(
                    "`@utility {name}`: the Tailwind CSS function `{other}()` is not implemented by diffpack's Tailwind compiler"
                )));
            }
        }
        i = end;
    }
    Ok(Some(out))
}

/// Reads a `( … )` group starting at the `(` at `open`. Returns the inner text
/// and the index just past the closing `)`.
fn read_parenthesized(text: &str, open: usize) -> Result<(String, usize), String> {
    let bytes = text.as_bytes();
    debug_assert_eq!(bytes[open], b'(');
    let mut depth = 0i32;
    let mut i = open;
    while i < bytes.len() {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Ok((text[open + 1..i].to_string(), i + 1));
                }
            }
            _ => {}
        }
        i += 1;
    }
    Err(format!("unbalanced parentheses in {text:?}"))
}

/// Resolves one `--value(a, b, …)` / `--modifier(a, b, …)` call: the first
/// argument the candidate's value satisfies wins; if none does, the call resolves
/// to nothing (`Ok(None)`) and its declaration is dropped.
fn resolve_value_call(
    args: &str,
    subject: Option<&str>,
    name: &str,
    function: &str,
    theme: &Theme,
) -> Result<Option<String>, Fail> {
    let Some(subject) = subject else {
        // A static `@utility` has no value, and a candidate with no `/…` has no
        // modifier: nothing can satisfy the call.
        return Ok(None);
    };
    for arg in split_top_level_commas(args) {
        if let Some(resolved) = resolve_value_arg(arg.trim(), subject, name, function, theme)? {
            return Ok(Some(resolved));
        }
    }
    Ok(None)
}

/// Resolves a single `--value(…)` argument against the candidate's value.
///
/// Three argument shapes, matching Tailwind:
/// * `--namespace-*` (optionally quoted) — a theme lookup: `stack-y-2` resolves
///   against `--namespace-2` and yields `var(--namespace-2)`.
/// * `[<type>]` — an ARBITRARY value: the candidate must be bracketed, and its
///   contents must be of `<type>` (`[*]` accepts anything).
/// * a bare data-type keyword (`integer`, `number`, …) — a BARE value of that type.
fn resolve_value_arg(
    arg: &str,
    subject: &str,
    name: &str,
    function: &str,
    theme: &Theme,
) -> Result<Option<String>, Fail> {
    let arg = unquote(arg);
    if let Some(namespace) = arg.strip_suffix("-*")
        && namespace.starts_with("--")
    {
        if subject.starts_with('[') {
            return Ok(None);
        }
        let token = format!("{namespace}-{subject}");
        return Ok(theme.contains(&token).then(|| format!("var({token})")));
    }
    if let Some(ty) = arg
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
    {
        let Some(inner) = arbitrary_value(subject) else {
            return Ok(None);
        };
        // `[length:var(--x)]` — an explicit data-type hint inside the candidate.
        let inner = match inner.split_once(':') {
            Some((hint, rest)) if is_value_data_type(hint) => {
                if hint != ty && ty != "*" {
                    return Ok(None);
                }
                rest.trim().to_string()
            }
            _ => inner,
        };
        return Ok(value_matches_data_type(&inner, ty, name, function)?.then_some(inner));
    }
    if subject.starts_with('[') {
        return Ok(None);
    }
    Ok(value_matches_data_type(subject, arg, name, function)?.then(|| subject.to_string()))
}

/// Strips one matching pair of surrounding quotes.
fn unquote(text: &str) -> &str {
    for quote in ['\'', '"'] {
        if text.len() >= 2 && text.starts_with(quote) && text.ends_with(quote) {
            return &text[1..text.len() - 1];
        }
    }
    text
}

/// The Tailwind data-type names this compiler understands (used to recognize an
/// explicit `[length:…]` hint inside a candidate's arbitrary value).
fn is_value_data_type(name: &str) -> bool {
    matches!(
        name,
        "*" | "any"
            | "integer"
            | "number"
            | "percentage"
            | "ratio"
            | "length"
            | "angle"
            | "time"
            | "color"
            | "url"
            | "string"
    )
}

/// Whether a CSS value satisfies a Tailwind `--value(…)` data type. An unknown
/// type name is an engine gap (hard error), never a silent non-match.
fn value_matches_data_type(
    value: &str,
    ty: &str,
    name: &str,
    function: &str,
) -> Result<bool, Fail> {
    let value = value.trim();
    if value.is_empty() {
        return Ok(false);
    }
    // Any math/variable expression is accepted for the numeric-ish types: its
    // result type cannot be decided statically, and Tailwind accepts it too.
    let computed = ["calc(", "var(", "min(", "max(", "clamp("]
        .iter()
        .any(|prefix| value.starts_with(prefix));
    Ok(match ty {
        "*" | "any" => true,
        "integer" => is_signed_integer(value),
        "number" => is_css_number(value),
        "percentage" => computed || value.strip_suffix('%').is_some_and(is_css_number),
        "ratio" => value
            .split_once('/')
            .is_some_and(|(a, b)| is_css_number(a.trim()) && is_css_number(b.trim())),
        "length" => computed || is_css_dimension(value, LENGTH_UNITS),
        "angle" => computed || is_css_dimension(value, ANGLE_UNITS),
        "time" => computed || is_css_dimension(value, TIME_UNITS),
        "color" => is_css_color(value),
        "url" => value.starts_with("url("),
        "string" => value.starts_with('"') || value.starts_with('\''),
        other => {
            return Err(Fail::Unsupported(format!(
                "`@utility {name}`: `{function}({other})` names a Tailwind data type that diffpack's Tailwind compiler does not implement"
            )));
        }
    })
}

const LENGTH_UNITS: &[&str] = &[
    "px", "rem", "em", "ex", "ch", "rex", "rch", "ic", "ric", "lh", "rlh", "vw", "vh", "vi", "vb",
    "vmin", "vmax", "svw", "svh", "svi", "svb", "svmin", "svmax", "lvw", "lvh", "lvi", "lvb",
    "lvmin", "lvmax", "dvw", "dvh", "dvi", "dvb", "dvmin", "dvmax", "cqw", "cqh", "cqi", "cqb",
    "cqmin", "cqmax", "cm", "mm", "q", "in", "pt", "pc",
];
const ANGLE_UNITS: &[&str] = &["deg", "grad", "rad", "turn"];
const TIME_UNITS: &[&str] = &["s", "ms"];

/// Splits a leading CSS number off a value, returning `(number, remainder)`.
fn split_number(value: &str) -> Option<(&str, &str)> {
    let bytes = value.as_bytes();
    let mut i = 0;
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        i += 1;
    }
    let mut digits = false;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
        digits = true;
    }
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
            digits = true;
        }
    }
    if !digits {
        return None;
    }
    Some((&value[..i], &value[i..]))
}

fn is_signed_integer(value: &str) -> bool {
    let digits = value.strip_prefix('-').unwrap_or(value);
    is_bare_integer(digits)
}

fn is_css_number(value: &str) -> bool {
    split_number(value).is_some_and(|(_, rest)| rest.is_empty())
}

fn is_css_dimension(value: &str, units: &[&str]) -> bool {
    let Some((number, unit)) = split_number(value) else {
        return false;
    };
    // A bare `0` is a valid length/angle/time; any other bare number is not.
    if unit.is_empty() {
        return number.trim_start_matches(['+', '-']).parse::<f64>() == Ok(0.0);
    }
    let unit = unit.to_ascii_lowercase();
    units.contains(&unit.as_str())
}

fn is_css_color(value: &str) -> bool {
    if let Some(hex) = value.strip_prefix('#') {
        return matches!(hex.len(), 3 | 4 | 6 | 8) && hex.bytes().all(|b| b.is_ascii_hexdigit());
    }
    for function in [
        "rgb(",
        "rgba(",
        "hsl(",
        "hsla(",
        "hwb(",
        "lab(",
        "lch(",
        "oklab(",
        "oklch(",
        "color(",
        "color-mix(",
        "var(",
        "light-dark(",
    ] {
        if value.starts_with(function) {
            return true;
        }
    }
    // A bare keyword (`red`, `currentColor`, `transparent`, …).
    !value.is_empty() && value.bytes().all(|b| b.is_ascii_alphabetic())
}

enum TopItem {
    Import,
    Layer {
        names: String,
        body: String,
    },
    Rule {
        selector: String,
        body: String,
    },
    /// `@utility <name> { … }` / `@utility <prefix>-* { … }` — an app-defined
    /// utility, registered before any candidate is rendered.
    Utility {
        name: String,
        body: String,
    },
    /// `@custom-variant <name> (<template>);` — a user-defined variant whose
    /// template (with `&` for the candidate selector) replaces the built-in
    /// meaning of `<name>:` for both utilities and `@apply` expansion.
    CustomVariant {
        name: String,
        template: String,
    },
    /// A top-level block passed through verbatim (`@keyframes`, `@media`,
    /// `@supports`, `@font-face`) — app CSS the compiler has no opinion on.
    Verbatim(String),
    /// `@theme [inline|reference|static] { --token: value; … }` — Tailwind v4
    /// design tokens declared inline in the app's stylesheet (create-next-app's
    /// default `globals.css` uses `@theme inline`). Its declarations extend the
    /// theme (so `font-sans`, `bg-foreground`, … resolve) and its referenced tokens
    /// are emitted into the theme-layer `:root` by [`Theme::render`]; the block
    /// itself produces no separate output here (only consumed so the parse does not
    /// error; the token text is pulled out separately by `extract_theme_blocks`).
    Theme,
}

/// Splits the entry into top-level items: `@import` statements, `@layer …{…}`
/// blocks, and plain style rules (passed through). Errors on any other
/// top-level at-rule.
/// Concatenate the bodies of every top-level `@theme [modifiers] { … }` block in
/// `css`, so the tokens can be merged into the theme (see [`compile_with_theme`]).
/// Malformed/unterminated blocks are skipped (the full parse in
/// [`parse_top_level`] reports the real error).
fn extract_theme_blocks(css: &str) -> String {
    let mut out = String::new();
    let mut search = 0;
    while let Some(rel) = css[search..].find("@theme") {
        let at = search + rel;
        let Some(brace_rel) = css[at..].find('{') else {
            break;
        };
        match read_braced(css, at + brace_rel) {
            Ok((body, end)) => {
                out.push_str(&body);
                out.push('\n');
                search = end;
            }
            Err(_) => break,
        }
    }
    out
}

/// Parses the body of a `@custom-variant <name> (<template>);` directive (the text
/// between the keyword and the `;`) into its `(name, &-rooted template)`.
fn parse_custom_variant(inner: &str) -> Result<(String, String), String> {
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
    Ok((name, template))
}

/// Collects every `@custom-variant` directive in a THEME source (as opposed to the
/// app's own stylesheet, which goes through [`parse_top_level`]). The theme source
/// is `tailwindcss/theme.css` plus the CSS a legacy JS config evaluates to, and the
/// latter carries the config's `darkMode` as a `dark` variant; the rest of that file
/// is token declarations this scan deliberately walks past.
fn scan_custom_variants(css: &str) -> Result<std::collections::BTreeMap<String, String>, String> {
    let mut out = std::collections::BTreeMap::new();
    let mut rest = css;
    while let Some(at) = rest.find("@custom-variant") {
        let body = &rest[at + "@custom-variant".len()..];
        let end = body.find(';').ok_or_else(|| {
            "malformed @custom-variant in the Tailwind theme source (no `;`)".to_string()
        })?;
        let (name, template) = parse_custom_variant(body[..end].trim())?;
        out.insert(name, template);
        rest = &body[end + 1..];
    }
    Ok(out)
}

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
            let end = css[i..]
                .find(';')
                .map(|rel| i + rel + 1)
                .unwrap_or(bytes.len());
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
        if css[i..].starts_with("@source") {
            // `@source "<glob>";` / `@source not "<glob>";` — extra files the
            // candidate scan covers. The scan happens before this compile (the
            // candidate set is an INPUT here), so the directive itself is a
            // consumed no-op; `bundler::tailwind_source_globs` reads them.
            let end = css[i..]
                .find(';')
                .map(|rel| i + rel + 1)
                .unwrap_or(bytes.len());
            i = end;
            continue;
        }
        if css[i..].starts_with("@config") {
            // `@config '<path>';` — a legacy (v3) JS config the bundler evaluates
            // (via node/jiti) into `@theme`/`@keyframes` tokens that are merged into
            // the theme source; here the directive itself is just consumed.
            let end = css[i..]
                .find(';')
                .map(|rel| i + rel + 1)
                .unwrap_or(bytes.len());
            i = end;
            continue;
        }
        if css[i..].starts_with("@tailwind") {
            // Legacy v3 `@tailwind base|components|utilities|screens|variants;`. The v4
            // assembler emits every one of those layers unconditionally, so each
            // directive is a consumed no-op marker here. An unknown layer keyword is a
            // hard error (never silently ignored).
            let end = css[i..].find(';').map(|rel| i + rel).unwrap_or(bytes.len());
            let keyword = css[i + "@tailwind".len()..end].trim();
            if !matches!(
                keyword,
                "base" | "components" | "utilities" | "screens" | "variants"
            ) {
                return Err(format!(
                    "unknown `@tailwind {keyword};` directive (expected base, components, utilities, screens, or variants)"
                ));
            }
            i = if end < bytes.len() { end + 1 } else { end };
            continue;
        }
        if css[i..].starts_with("@theme") {
            // `@theme [inline|reference|static] { … }` — any modifiers between the
            // keyword and the `{` are accepted and ignored (the tokens are merged
            // into the theme regardless; see `extract_theme_blocks`).
            let brace = css[i..]
                .find('{')
                .ok_or_else(|| "malformed @theme (no `{`)".to_string())?;
            let (_body, end) = read_braced(css, i + brace)?;
            items.push(TopItem::Theme);
            i = end;
            continue;
        }
        if css[i..].starts_with("@custom-variant") {
            let end = css[i..]
                .find(';')
                .map(|rel| i + rel)
                .ok_or_else(|| "malformed @custom-variant (no `;`)".to_string())?;
            let inner = css[i + "@custom-variant".len()..end].trim();
            let (name, template) = parse_custom_variant(inner)?;
            items.push(TopItem::CustomVariant { name, template });
            i = end + 1;
            continue;
        }
        if css[i..].starts_with("@utility") {
            let brace = css[i..]
                .find('{')
                .ok_or_else(|| "malformed @utility (no `{`)".to_string())?;
            let name = css[i + "@utility".len()..i + brace].trim().to_string();
            let (body, end) = read_braced(css, i + brace)?;
            if name.is_empty() {
                return Err("malformed @utility (no name)".to_string());
            }
            items.push(TopItem::Utility { name, body });
            i = end;
            continue;
        }
        // Plain CSS at-rules with a block: reproduced verbatim. These carry no
        // Tailwind semantics, so the compiler passes them straight through to
        // the emitted stylesheet.
        if VERBATIM_AT_RULES.iter().any(|at| {
            css[i..].starts_with(at)
                && css
                    .as_bytes()
                    .get(i + at.len())
                    .is_none_or(|b| !is_ident_byte(*b))
        }) {
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
        let selector = css[i..i + brace]
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
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
    utils: &CustomUtilities,
    dialect: Dialect,
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
                // `@apply bg-black!` carries the same important marker a class
                // attribute would; it belongs to the candidate, not the utility.
                let (apply_base, important) = split_important(apply_base);
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
                // An app-defined `@utility` overrides the built-in of the same
                // name, in `@apply` exactly as in a class attribute.
                let decls = match utils.lookup(apply_base) {
                    Some(found) => {
                        let expanded = expand_utility_body(
                            found.body,
                            apply_base,
                            found.value,
                            found.modifier,
                            theme,
                            tw_props,
                            dialect,
                            "",
                        )
                        .map_err(|fail| fail.into_apply_error(class))?;
                        if !expanded.nested.is_empty() {
                            return Err(format!(
                                "`@apply {class}`: that `@utility` has nested rules, which cannot be flattened into `{}`",
                                rule.selector
                            ));
                        }
                        if expanded.decls.is_empty() {
                            return Err(Fail::Invalid.into_apply_error(class));
                        }
                        expanded.decls
                    }
                    None => {
                        generate_utility(apply_base, class, theme, tw_props, dialect)
                            .map_err(|fail| fail.into_apply_error(class))?
                            .decls
                    }
                };
                for (prop, value) in decls {
                    let decl = format!("{prop}:{}", with_important(&value, important));
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
    fn detects_tailwind_v3_entry_and_gate() {
        assert!(is_tailwind_v3_entry(
            "@tailwind base;\n@tailwind components;\n@tailwind utilities;\n"
        ));
        assert!(is_tailwind_v3_entry("@tailwind utilities;"));
        assert!(!is_tailwind_v3_entry(".a{color:red}"));
        assert!(!is_tailwind_v3_entry("@import 'tailwindcss';")); // v4, not v3
        // The shared gate accepts both dialects.
        assert!(needs_native_tailwind_compile("@import 'tailwindcss';"));
        assert!(needs_native_tailwind_compile(
            "@tailwind base;\n@tailwind utilities;"
        ));
        assert!(!needs_native_tailwind_compile(".a{color:red}"));
    }

    #[test]
    fn v3_tailwind_directives_parse_as_consumed_no_ops() {
        // A v3 entry parses (the directives are consumed markers) and compiles a scanned
        // utility from the vendored v4 base theme.
        let css = "@tailwind base;\n@tailwind components;\n@tailwind utilities;\n";
        let out = compile(css, &candidates(&["underline"])).unwrap();
        assert!(!out.contains("@tailwind"), "directives consumed: {out}");
        assert!(
            out.contains("underline") && out.contains("text-decoration"),
            "{out}"
        );
        // An unknown layer keyword is a hard error, not a silent skip.
        let err = compile("@tailwind bogus;\n", &candidates(&[])).unwrap_err();
        assert!(err.contains("@tailwind bogus"), "{err}");
    }

    #[test]
    fn scans_class_candidates_from_every_context() {
        let mut out = BTreeSet::new();
        scan_class_candidates(r#"<div className="p-2 flex gap-2" />"#, &mut out);
        scan_class_candidates(r#"className={`px-2 py-1 font-extrabold`}"#, &mut out);
        scan_class_candidates(
            r#"activeProps={{ className: 'text-black font-bold' }}"#,
            &mut out,
        );
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
        assert!(
            !out.contains("prefers-color-scheme"),
            "no media dark: {out}"
        );
        assert!(
            out.contains("@keyframes pop"),
            "verbatim keyframes survive: {out}"
        );
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
            ".text-lg{font-size:var(--text-lg);line-height:var(--tw-leading, var(--text-lg--line-height))}"
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
            "@media (hover: hover){.hover\\:text-blue-600:hover{color:var(--color-blue-600)}}"
        ));
        assert!(out.contains(
            "@media (prefers-color-scheme: dark){.dark\\:bg-gray-700{background-color:var(--color-gray-700)}}"
        ));
    }

    #[test]
    fn space_y_emits_reverse_variable_and_child_selector() {
        let out = compile("@import 'tailwindcss';", &candidates(&["space-y-2"])).unwrap();
        assert!(out.contains(":where(.space-y-2 > :not(:last-child)){--tw-space-y-reverse:0;margin-block-start:calc(calc(var(--spacing) * 2) * var(--tw-space-y-reverse));margin-block-end:calc(calc(var(--spacing) * 2) * calc(1 - var(--tw-space-y-reverse)))}"));
        assert!(out.contains(
            "@property --tw-space-y-reverse{syntax:\"*\";inherits:false;initial-value:0}"
        ));
        // Fractional steps are part of the same scale.
        let out = compile("@import 'tailwindcss';", &candidates(&["space-y-1.5"])).unwrap();
        assert!(out.contains(":where(.space-y-1\\.5 > :not(:last-child))"));
        assert!(out.contains("calc(var(--spacing) * 1.5)"));
    }

    #[test]
    fn border_side_emits_style_and_width() {
        let out = compile("@import 'tailwindcss';", &candidates(&["border-b"])).unwrap();
        assert!(out.contains(
            ".border-b{border-bottom-style:var(--tw-border-style);border-bottom-width:1px}"
        ));
        assert!(out.contains(
            "@property --tw-border-style{syntax:\"*\";inherits:false;initial-value:solid}"
        ));
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
            "@media (width >= 48rem){.md\\:border-t-0{border-top-style:var(--tw-border-style);border-top-width:0}}"
        ));
        assert!(out.contains(".border-l-amber-500{border-left-color:var(--color-amber-500)}"));
        assert!(out.contains(".border-transparent{border-color:transparent}"));
        assert!(
            out.contains(".border-\\[color\\:var\\(--border\\)\\]{border-color:var(--border)}")
        );
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
        assert!(out.contains(".top-1\\/2{top:calc(1 / 2 * 100%)}"));
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
                "w-full",
                "w-px",
                "w-screen",
                "w-0.5",
                "h-screen",
                "h-1.5",
                "h-px",
                "min-w-0",
                "min-w-[10rem]",
                "min-h-0",
                "max-w-sm",
                "max-w-[18rem]",
                "max-w-screen-md",
                "size-5",
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
        assert!(
            out.contains(".size-5{width:calc(var(--spacing) * 5);height:calc(var(--spacing) * 5)}")
        );
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
        assert!(
            out.contains("@property --tw-translate-x{syntax:\"*\";inherits:false;initial-value:0}")
        );
        assert!(out.contains("--tw-translate-z:0"));
    }

    #[test]
    fn margin_auto_negatives_and_all_sides() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&[
                "m-auto", "ml-auto", "mx-2", "my-1.5", "mr-1", "mb-4", "-mt-2",
            ]),
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
            &candidates(&[
                "gap-x-3",
                "gap-y-2",
                "gap-0.5",
                "grid-cols-2",
                "grid-cols-[auto_1fr]",
            ]),
        )
        .unwrap();
        assert!(out.contains(".gap-x-3{column-gap:calc(var(--spacing) * 3)}"));
        assert!(out.contains(".gap-y-2{row-gap:calc(var(--spacing) * 2)}"));
        assert!(out.contains(".gap-0\\.5{gap:calc(var(--spacing) * 0.5)}"));
        assert!(out.contains(".grid-cols-2{grid-template-columns:repeat(2, minmax(0, 1fr))}"));
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
        assert!(out.contains(".bg-\\[color\\:var\\(--panel\\)\\]{background-color:var(--panel)}"));
        assert!(out.contains(
            ".bg-amber-500\\/10{background-color:color-mix(in oklab, var(--color-amber-500) 10%, transparent);background-color:color-mix(in srgb, oklch(76.9% 0.188 70.08) 10%, transparent)}"
        ));
        assert!(out.contains(".bg-transparent{background-color:transparent}"));
        assert!(out.contains(".text-\\[color\\:var\\(--muted\\)\\]{color:var(--muted)}"));
        assert!(out.contains(".text-\\[11px\\]{font-size:11px}"));
        assert!(
            out.contains(".accent-\\[color\\:var\\(--accent\\)\\]{accent-color:var(--accent)}")
        );
        assert!(out.contains("--color-amber-500:"));
    }

    // The exact Tailwind surface the UNMODIFIED create-next-app default template
    // uses (its `globals.css` `@theme inline` block + `app/page.tsx` classes) — a
    // real-world app that must build. Regression guard for that support.
    #[test]
    fn create_next_app_default_tailwind_surface() {
        let css = "@import \"tailwindcss\";\n\
                   :root { --background: #ffffff; --foreground: #171717; }\n\
                   @theme inline {\n\
                     --color-background: var(--background);\n\
                     --color-foreground: var(--foreground);\n\
                     --font-sans: var(--font-geist-sans);\n\
                   }\n\
                   body { background: var(--background); }";
        let out = compile(
            css,
            &candidates(&[
                "bg-[#383838]",
                "hover:bg-[#383838]",
                "dark:hover:bg-[#ccc]",
                "bg-black/[.04]",
                "border-black/[.08]",
                "dark:border-white/[.145]",
                "bg-foreground",
                "text-background",
                "font-sans",
                "antialiased",
                "border-solid",
            ]),
        )
        .unwrap();
        // Bare arbitrary hex color (with and without variants).
        assert!(out.contains(".bg-\\[\\#383838\\]{background-color:#383838}"));
        assert!(out.contains(".hover\\:bg-\\[\\#383838\\]:hover{background-color:#383838}"));
        // Arbitrary opacity modifier: `.04` -> 4%, `.08` -> 8%, `.145` -> 14.5%.
        assert!(out.contains(
            ".bg-black\\/\\[\\.04\\]{background-color:color-mix(in oklab, var(--color-black) 4%, transparent);background-color:color-mix(in srgb, #000 4%, transparent)}"
        ));
        assert!(out.contains(
            ".border-black\\/\\[\\.08\\]{border-color:color-mix(in oklab, var(--color-black) 8%, transparent)}"
        ));
        assert!(out.contains("14.5%"));
        // `@theme inline` tokens resolve for utilities + emit their :root vars.
        assert!(out.contains(".bg-foreground{background-color:var(--color-foreground)}"));
        assert!(out.contains(".font-sans{font-family:var(--font-sans)}"));
        assert!(out.contains("--color-foreground:var(--foreground)"));
        assert!(out.contains("--font-sans:var(--font-geist-sans)"));
        // Static keyword utilities the template relies on.
        assert!(out.contains("-webkit-font-smoothing:antialiased"));
        assert!(out.contains(".border-solid{--tw-border-style:solid;border-style:solid}"));
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
        assert!(
            out.contains(".truncate{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}")
        );
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
            &candidates(&[
                "shadow-lg",
                "ring-2",
                "ring-inset",
                "outline",
                "outline-none",
            ]),
        )
        .unwrap();
        assert!(out.contains(
            ".shadow-lg{--tw-shadow:0 10px 15px -3px var(--tw-shadow-color, rgb(0 0 0 / 0.1)), 0 4px 6px -4px var(--tw-shadow-color, rgb(0 0 0 / 0.1));box-shadow:var(--tw-inset-shadow), var(--tw-inset-ring-shadow), var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow)}"
        ));
        assert!(out.contains(
            ".ring-2{--tw-ring-shadow:var(--tw-ring-inset,) 0 0 0 calc(2px + var(--tw-ring-offset-width)) var(--tw-ring-color, currentcolor);box-shadow:"
        ));
        assert!(out.contains(".ring-inset{--tw-ring-inset:inset}"));
        assert!(out.contains(".outline{outline-style:var(--tw-outline-style);outline-width:1px}"));
        assert!(out.contains(".outline-none{--tw-outline-style:none;outline-style:none}"));
        assert!(out.contains(
            "@property --tw-ring-offset-color{syntax:\"*\";inherits:false;initial-value:#fff}"
        ));
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
            ".transition-colors{transition-property:color, background-color, border-color, outline-color, text-decoration-color, fill, stroke, --tw-gradient-from, --tw-gradient-via, --tw-gradient-to;transition-timing-function:var(--tw-ease, var(--default-transition-timing-function));transition-duration:var(--tw-duration, var(--default-transition-duration))}"
        ));
        assert!(out.contains(".transition-opacity{transition-property:opacity;"));
        assert!(out.contains(".cursor-pointer{cursor:pointer}"));
        assert!(out.contains(".cursor-col-resize{cursor:col-resize}"));
        assert!(out.contains(".select-none{-webkit-user-select:none;user-select:none}"));
        assert!(out.contains(".pointer-events-none{pointer-events:none}"));
        assert!(out.contains(".opacity-45{opacity:45%}"));
        assert!(out.contains(".opacity-0{opacity:0%}"));
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
        assert!(out.contains(".rounded-full{border-radius:calc(infinity * 1px)}"));
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
        assert!(
            out.contains(".focus\\:outline-none:focus{--tw-outline-style:none;outline-style:none}")
        );
        assert!(out.contains(".focus-visible\\:ring-2:focus-visible{--tw-ring-shadow:"));
        assert!(out.contains(".disabled\\:opacity-50:disabled{opacity:50%}"));
        assert!(out.contains(
            "@media (hover: hover){.disabled\\:hover\\:bg-transparent:disabled:hover{background-color:transparent}}"
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
        assert!(
            out.contains(".before\\:absolute:before{content:var(--tw-content);position:absolute}")
        );
        assert!(out.contains(
            ".before\\:-left-1:before{content:var(--tw-content);left:calc(var(--spacing) * -1)}"
        ));
        assert!(out.contains(
            ".before\\:content-\\[\\'\\'\\]:before{--tw-content:'';content:var(--tw-content)}"
        ));
        assert!(out.contains(
            ".backdrop\\:bg-black\\/40::backdrop{background-color:color-mix(in oklab, var(--color-black) 40%, transparent);background-color:color-mix(in srgb, #000 40%, transparent)}"
        ));
        assert!(out.contains(
            "@media (hover: hover){.group-hover\\:opacity-100:is(:where(.group):hover *){opacity:100%}}"
        ));
        // `group` itself is a marker class: no rule, no error.
        assert!(!out.contains(".group{"));
        assert!(
            out.contains("@property --tw-content{syntax:\"*\";inherits:false;initial-value:\"\"}")
        );
    }

    #[test]
    fn breakpoint_variants_use_theme_widths() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&["md:flex-row", "sm:px-2", "lg:grid-cols-3"]),
        )
        .unwrap();
        assert!(out.contains("@media (width >= 48rem){.md\\:flex-row{flex-direction:row}}"));
        assert!(out.contains(
            "@media (width >= 40rem){.sm\\:px-2{padding-inline:calc(var(--spacing) * 2)}}"
        ));
        assert!(out.contains(
            "@media (width >= 64rem){.lg\\:grid-cols-3{grid-template-columns:repeat(3, minmax(0, 1fr))}}"
        ));
        // Breakpoint blocks come in ascending width order.
        let sm = out.find("width >= 40rem").unwrap();
        let md = out.find("width >= 48rem").unwrap();
        let lg = out.find("width >= 64rem").unwrap();
        assert!(sm < md && md < lg);
    }

    #[test]
    fn plain_user_rules_pass_through_after_utilities() {
        let css = "@import 'tailwindcss';\n\
            :root { --bg: #ffffff; color-scheme: light; }\n\
            .no-scrollbar { scrollbar-width: none; }\n\
            .no-scrollbar::-webkit-scrollbar { display: none; }\n\
            .markpad-preview h1 { font-size: 1.75rem; }\n";
        let out = compile(
            css,
            &candidates(&["flex", "no-scrollbar", "markpad-preview"]),
        )
        .unwrap();
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
        assert!(
            out.contains(
                "html, body{color:var(--color-gray-900);background-color:var(--color-gray-50)}"
            ) || out.contains(
                "html, body{background-color:var(--color-gray-50);color:var(--color-gray-900)}"
            )
        );
        // The dark applies land in a dark media rule with the same selector.
        assert!(out.contains("@media (prefers-color-scheme: dark){html, body{"));
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
        let err = compile("@import 'tailwindcss';", &candidates(&["bg-plaid-500"])).unwrap_err();
        assert!(
            err.contains("bg-plaid-500"),
            "error must name the token: {err}"
        );

        let err = compile("@import 'tailwindcss';", &candidates(&["text-gray-1000"])).unwrap_err();
        assert!(
            err.contains("text-gray-1000"),
            "error must name the token: {err}"
        );

        // All failures are reported together, not one at a time.
        let err = compile(
            "@import 'tailwindcss';",
            &candidates(&["text-gray-1000", "bg-plaid-500"]),
        )
        .unwrap_err();
        assert!(err.contains("text-gray-1000") && err.contains("bg-plaid-500"));
    }

    #[test]
    fn unknown_variant_is_a_hard_error() {
        // A genuinely unimplemented variant still hard-errors (naming it), never
        // silently drops. (`aria-checked`, `group-focus`, `supports-*`, etc. are
        // now supported; a group/peer ARIA state is a real remaining gap.)
        let err = compile(
            "@import 'tailwindcss';",
            &candidates(&["group-aria-checked:flex"]),
        )
        .unwrap_err();
        assert!(
            err.contains("group-aria-checked"),
            "error must name the variant: {err}"
        );
    }

    #[test]
    fn gradient_directions_and_color_stops() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&[
                "bg-gradient-to-br",
                "bg-linear-to-t",
                "from-rose-50",
                "via-indigo-50",
                "to-amber-50",
                "from-10%",
            ]),
        )
        .unwrap();
        assert!(out.contains(
            ".bg-gradient-to-br{--tw-gradient-position:to bottom right in oklab;background-image:linear-gradient(var(--tw-gradient-stops))}"
        ));
        assert!(out.contains(
            ".bg-linear-to-t{--tw-gradient-position:to top in oklab;background-image:linear-gradient(var(--tw-gradient-stops))}"
        ));
        assert!(out.contains(
            ".from-rose-50{--tw-gradient-from:var(--color-rose-50);--tw-gradient-stops:"
        ));
        assert!(out.contains(
            ".via-indigo-50{--tw-gradient-via:var(--color-indigo-50);--tw-gradient-via-stops:var(--tw-gradient-position), var(--tw-gradient-from) var(--tw-gradient-from-position), var(--tw-gradient-via) var(--tw-gradient-via-position), var(--tw-gradient-to) var(--tw-gradient-to-position);--tw-gradient-stops:var(--tw-gradient-via-stops)}"
        ));
        assert!(
            out.contains(
                ".to-amber-50{--tw-gradient-to:var(--color-amber-50);--tw-gradient-stops:"
            )
        );
        assert!(out.contains(".from-10\\%{--tw-gradient-from-position:10%}"));
        // The whole gradient property group registers.
        assert!(out.contains("@property --tw-gradient-position{syntax:\"*\";inherits:false}"));
        assert!(out.contains(
            "@property --tw-gradient-from{syntax:\"<color>\";inherits:false;initial-value:#0000}"
        ));
        assert!(out.contains(
            "@property --tw-gradient-via-position{syntax:\"<length-percentage>\";inherits:false;initial-value:50%}"
        ));
    }

    #[test]
    fn important_marker_prefix_and_suffix() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&[
                "!text-white",
                "hover:!bg-indigo-600",
                "!border-0",
                "bg-black!",
            ]),
        )
        .unwrap();
        assert!(out.contains(".\\!text-white{color:var(--color-white)!important}"));
        assert!(out.contains(
            ".hover\\:\\!bg-indigo-600:hover{background-color:var(--color-indigo-600)!important}"
        ));
        assert!(out.contains(
            ".\\!border-0{border-style:var(--tw-border-style)!important;border-width:0!important}"
        ));
        assert!(out.contains(".bg-black\\!{background-color:var(--color-black)!important}"));
    }

    /// The important marker belongs to the CANDIDATE, so `@apply` carries it too
    /// — in a plain rule, in a `@layer base` rule, and inside an `@utility` body.
    /// It was only ever stripped on the scanned-class path, so `@apply bg-black!`
    /// (the v4 spelling, which real apps use) reported the utility as invalid.
    #[test]
    fn the_important_marker_is_stripped_on_the_apply_path_too() {
        let out = compile(
            "@import 'tailwindcss';\n\
             @utility boxed {\n  @apply border-0!;\n}\n\
             @layer base {\n  .layered { @apply bg-black!; }\n}\n\
             .suffix { @apply bg-black!; }\n\
             .prefix { @apply !bg-black; }\n\
             .plain { @apply bg-black; }\n",
            &candidates(&["boxed"]),
        )
        .unwrap();
        assert!(
            out.contains(".suffix{background-color:var(--color-black)!important}"),
            "{out}"
        );
        assert!(
            out.contains(".prefix{background-color:var(--color-black)!important}"),
            "{out}"
        );
        // Unmarked `@apply` is untouched: the marker is not applied wholesale.
        assert!(
            out.contains(".plain{background-color:var(--color-black)}"),
            "{out}"
        );
        assert!(
            out.contains(".layered{background-color:var(--color-black)!important}"),
            "{out}"
        );
        assert!(
            out.contains("border-style:var(--tw-border-style)!important"),
            "an `@apply x!` inside an `@utility` body marks its declarations: {out}"
        );
    }

    #[test]
    fn shadow_bare_none_and_arbitrary_wrap_colors() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&[
                "shadow",
                "shadow-none",
                "shadow-[0_-2px_12px_-4px_rgba(0,0,0,0.08)]",
            ]),
        )
        .unwrap();
        // Bare `shadow` is the scale's `sm` entry.
        assert!(out.contains(
            ".shadow{--tw-shadow:0 1px 3px 0 var(--tw-shadow-color, rgb(0 0 0 / 0.1)), 0 1px 2px -1px var(--tw-shadow-color, rgb(0 0 0 / 0.1));box-shadow:"
        ));
        assert!(out.contains(".shadow-none{--tw-shadow:0 0 #0000;box-shadow:"));
        assert!(out.contains(
            "{--tw-shadow:0 -2px 12px -4px var(--tw-shadow-color, rgba(0,0,0,0.08));box-shadow:"
        ));
    }

    #[test]
    fn drop_shadow_sized_and_bare() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&["drop-shadow", "drop-shadow-md"]),
        )
        .unwrap();
        // Sized: color wrapped for --tw-drop-shadow-color, plain keeps the var.
        assert!(out.contains(
            ".drop-shadow-md{--tw-drop-shadow-size:drop-shadow(0 3px 3px var(--tw-drop-shadow-color,rgb(0 0 0 / 0.12)));--tw-drop-shadow:drop-shadow(var(--drop-shadow-md));filter:var(--tw-blur,)"
        ));
        // Bare: the two default layers, each inlined.
        assert!(out.contains(
            ".drop-shadow{--tw-drop-shadow-size:drop-shadow(0 1px 2px var(--tw-drop-shadow-color,rgb(0 0 0 / 0.1))) drop-shadow(0 1px 1px var(--tw-drop-shadow-color,rgb(0 0 0 / 0.06)));--tw-drop-shadow:drop-shadow(0 1px 2px rgb(0 0 0 / 0.1)) drop-shadow(0 1px 1px rgb(0 0 0 / 0.06));filter:"
        ));
        assert!(out.contains("@property --tw-drop-shadow-alpha{syntax:\"<percentage>\";inherits:false;initial-value:100%}"));
    }

    #[test]
    fn backdrop_blur_bare_sized_and_arbitrary() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&["backdrop-blur", "backdrop-blur-md", "backdrop-blur-[2px]"]),
        )
        .unwrap();
        assert!(out.contains(".backdrop-blur{--tw-backdrop-blur:blur(8px);-webkit-backdrop-filter:var(--tw-backdrop-blur,)"));
        assert!(out.contains(".backdrop-blur-md{--tw-backdrop-blur:blur(var(--blur-md));"));
        assert!(out.contains(".backdrop-blur-\\[2px\\]{--tw-backdrop-blur:blur(2px);"));
        assert!(out.contains("@property --tw-backdrop-sepia{syntax:\"*\";inherits:false}"));
    }

    #[test]
    fn duration_scale_and_arbitrary() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&["duration-300", "duration-[2s]"]),
        )
        .unwrap();
        assert!(out.contains(".duration-300{--tw-duration:300ms;transition-duration:300ms}"));
        assert!(out.contains(".duration-\\[2s\\]{--tw-duration:2s;transition-duration:2s}"));
        assert!(out.contains("@property --tw-duration{syntax:\"*\";inherits:false}"));
    }

    #[test]
    fn animate_theme_scale_emits_keyframes() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&["animate-pulse", "animate-spin"]),
        )
        .unwrap();
        assert!(out.contains(".animate-pulse{animation:var(--animate-pulse)}"));
        assert!(out.contains(".animate-spin{animation:var(--animate-spin)}"));
        // Theme tokens and their keyframes are both emitted.
        assert!(out.contains("--animate-pulse:pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite"));
        assert!(out.contains("@keyframes pulse{50%{opacity:0.5}}"));
        assert!(out.contains("@keyframes spin{to{transform:rotate(360deg)}}"));
        // A name with no theme token resolves against nothing — like the
        // reference, no rule and no error.
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&["animate-bounce-in", "flex"]),
        )
        .unwrap();
        assert!(!out.contains("animate-bounce-in"));
    }

    #[test]
    fn scale_and_transform_register_their_properties() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&["hover:scale-110", "transform"]),
        )
        .unwrap();
        assert!(out.contains(
            ".hover\\:scale-110:hover{--tw-scale-x:110%;--tw-scale-y:110%;--tw-scale-z:110%;scale:var(--tw-scale-x) var(--tw-scale-y)}"
        ));
        assert!(out.contains(
            ".transform{transform:var(--tw-rotate-x,) var(--tw-rotate-y,) var(--tw-rotate-z,) var(--tw-skew-x,) var(--tw-skew-y,)}"
        ));
        assert!(
            out.contains("@property --tw-scale-x{syntax:\"*\";inherits:false;initial-value:1}")
        );
        assert!(out.contains("@property --tw-rotate-x{syntax:\"*\";inherits:false}"));
        assert!(out.contains("@property --tw-skew-y{syntax:\"*\";inherits:false}"));
    }

    #[test]
    fn rounded_sides_and_corners() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&[
                "rounded-t-2xl",
                "rounded-b-2xl",
                "rounded-tl-sm",
                "rounded-e-full",
                "rounded-r",
            ]),
        )
        .unwrap();
        assert!(out.contains(
            ".rounded-t-2xl{border-top-left-radius:var(--radius-2xl);border-top-right-radius:var(--radius-2xl)}"
        ));
        assert!(out.contains(
            ".rounded-b-2xl{border-bottom-right-radius:var(--radius-2xl);border-bottom-left-radius:var(--radius-2xl)}"
        ));
        assert!(out.contains(".rounded-tl-sm{border-top-left-radius:var(--radius-sm)}"));
        assert!(out.contains(
            ".rounded-e-full{border-start-end-radius:calc(infinity * 1px);border-end-end-radius:calc(infinity * 1px)}"
        ));
        assert!(out.contains(
            ".rounded-r{border-top-right-radius:0.25rem;border-bottom-right-radius:0.25rem}"
        ));
        // Sides sort after whole-box radii, `t` before `b` (Tailwind's order).
        let t = out.find(".rounded-t-2xl").unwrap();
        let b = out.find(".rounded-b-2xl").unwrap();
        assert!(t < b);
    }

    #[test]
    fn vertical_align_family() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&["align-middle", "align-text-bottom"]),
        )
        .unwrap();
        assert!(out.contains(".align-middle{vertical-align:middle}"));
        assert!(out.contains(".align-text-bottom{vertical-align:text-bottom}"));
    }

    #[test]
    fn aspect_family_and_invalid_value_skips() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&[
                "aspect-square",
                "aspect-video",
                "aspect-16/9",
                "aspect-ratio-1",
            ]),
        )
        .unwrap();
        assert!(out.contains(".aspect-square{aspect-ratio:1 / 1}"));
        assert!(out.contains(".aspect-video{aspect-ratio:var(--aspect-video)}"));
        assert!(out.contains(".aspect-16\\/9{aspect-ratio:16/9}"));
        // `aspect-ratio-1` resolves against nothing: no rule, no error.
        assert!(!out.contains("aspect-ratio-1"));
    }

    #[test]
    fn viewport_unit_sizing() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&["min-h-dvh", "h-svh", "w-dvw", "max-h-lvh"]),
        )
        .unwrap();
        assert!(out.contains(".min-h-dvh{min-height:100dvh}"));
        assert!(out.contains(".h-svh{height:100svh}"));
        assert!(out.contains(".w-dvw{width:100dvw}"));
        assert!(out.contains(".max-h-lvh{max-height:100lvh}"));
    }

    #[test]
    fn z_index_and_spacing_arbitrary_values() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&["z-[100]", "gap-[2px]", "p-[7px]", "-m-[3px]"]),
        )
        .unwrap();
        assert!(out.contains(".z-\\[100\\]{z-index:100}"));
        assert!(out.contains(".gap-\\[2px\\]{gap:2px}"));
        assert!(out.contains(".p-\\[7px\\]{padding:7px}"));
        assert!(out.contains(".-m-\\[3px\\]{margin:calc(3px * -1)}"));
    }

    #[test]
    fn display_table_box_sizing_and_whitespace_keywords() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&["table", "table-cell", "box-border", "whitespace-pre-line"]),
        )
        .unwrap();
        assert!(out.contains(".table{display:table}"));
        assert!(out.contains(".table-cell{display:table-cell}"));
        assert!(out.contains(".box-border{box-sizing:border-box}"));
        assert!(out.contains(".whitespace-pre-line{white-space:pre-line}"));
    }

    #[test]
    fn malformed_variant_and_fragment_candidates_are_skipped() {
        // `!dark:` is not a possible variant (`!` only marks the utility):
        // Tailwind rejects the candidate outright, generating nothing.
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&[
                "hover:!dark:bg-rose-400",
                "focus:!dark:ring-rose-300",
                "flex",
            ]),
        )
        .unwrap();
        assert!(!out.contains("dark:bg-rose-400"));
        assert!(!out.contains("dark:ring-rose-300"));
        // A template-literal fragment (`grid-cols-${n}`) scans as `grid-cols-`
        // and is likewise not a candidate Tailwind accepts.
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&["grid-cols-", "flex"]),
        )
        .unwrap();
        assert!(!out.contains("grid-cols-"));
        // An unknown `transition-` value resolves against nothing (the typo
        // `transition-color`), but the arbitrary form stays an engine gap.
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&["transition-color", "flex"]),
        )
        .unwrap();
        assert!(!out.contains("transition-color"));
        compile(
            "@import 'tailwindcss';",
            &candidates(&["transition-[height]"]),
        )
        .unwrap_err();
    }

    #[test]
    fn scans_call_arguments_boolean_guards_and_comparisons() {
        let mut out = BTreeSet::new();
        scan_class_candidates(
            r#"
            <div className={clsx(
              'relative aspect-square border',
              'transition-all duration-300',
              phase !== 'finished' && territory === "R" ? 'bg-rose-100' : 'bg-indigo-100',
              legal.has(`${x},${y}`) && 'hover:bg-emerald-200/40',
            )} />
            "#,
            &mut out,
        );
        assert!(out.contains("relative"));
        assert!(out.contains("aspect-square"));
        assert!(out.contains("transition-all"));
        assert!(out.contains("duration-300"));
        assert!(out.contains("bg-rose-100"));
        assert!(out.contains("bg-indigo-100"));
        assert!(out.contains("hover:bg-emerald-200/40"));
        // Compared operands are not class lists.
        assert!(!out.contains("finished"));
        assert!(!out.contains("R"));
    }

    #[test]
    fn scans_class_suffixed_props_and_object_maps() {
        let mut out = BTreeSet::new();
        scan_class_candidates(
            r#"
            const dirs = [
              { dir: 'top', btnClass: 'absolute left-1/2 -translate-x-1/2' },
            ];
            <WallButton divClass="h-[60%] rounded" />
            "#,
            &mut out,
        );
        assert!(out.contains("absolute"));
        assert!(out.contains("left-1/2"));
        assert!(out.contains("-translate-x-1/2"));
        assert!(out.contains("h-[60%]"));
        assert!(out.contains("rounded"));
        // The non-class object value is a candidate only in the scanner's
        // token sense; it never becomes CSS (unrecognized root).
        assert!(!out.contains("dir"));
    }

    #[test]
    fn resolves_identifier_bindings_across_files() {
        let colors = r#"
            export const COLOR = {
              warning: [
                'bg-amber-500 text-white border-amber-400',
                'dark:bg-amber-600 dark:border-amber-500',
              ].join(' '),
            }
        "#;
        let button = r#"
            import { COLOR } from '@/lib/colors'
            const variantClass = active ? COLOR.success : (variant && COLOR[variant]) || COLOR.neutral
            export const Button = () => <button className={clsx('px-3', variantClass)} />
        "#;
        let mut out = BTreeSet::new();
        scan_class_candidates_multi(&[colors, button], &mut out);
        assert!(out.contains("px-3"));
        assert!(out.contains("bg-amber-500"));
        assert!(out.contains("text-white"));
        assert!(out.contains("dark:border-amber-500"));
    }

    /// The identifier resolution is driven by a one-pass binding INDEX rather than by
    /// re-scanning every file for every name. These lock the index's contract.
    #[test]
    fn binding_index_finds_every_declaration_form_the_name_rescan_found() {
        let source = r#"
            const base = "px-3 py-1";
            let hovered = 'hover:bg-slate-100';
            var legacy = `text-sm`;
            const { destructured } = props;
            const notAssigned;
            for (const item of items) {}
            const arrow = () => "rounded-lg";
            const shorthand => never;
            const eq == never;
        "#;
        let index = binding_initializers(source);
        let by_name = |name: &str| {
            index
                .iter()
                .filter(|(n, _)| *n == name)
                .map(|(_, init)| init.trim())
                .collect::<Vec<_>>()
        };
        assert_eq!(by_name("base"), vec!["\"px-3 py-1\""]);
        assert_eq!(by_name("hovered"), vec!["'hover:bg-slate-100'"]);
        assert_eq!(by_name("legacy"), vec!["`text-sm`"]);
        // A destructuring pattern binds no single name this scan can resolve.
        assert!(
            by_name("destructured").is_empty(),
            "destructuring is not a name binding"
        );
        assert!(by_name("props").is_empty());
        // `const notAssigned;` and `for (const item of …)` have no `=`.
        assert!(by_name("notAssigned").is_empty());
        assert!(by_name("item").is_empty());
        // An arrow-function initializer IS indexed — returning a class string from one
        // is a real pattern, and the fixpoint decides eligibility, not this scan.
        assert_eq!(by_name("arrow"), vec!["() => \"rounded-lg\""]);
        // A bare `=>` / `==` after the name is not an assignment at all.
        assert!(by_name("shorthand").is_empty(), "`=>` is not an assignment");
        assert!(by_name("eq").is_empty(), "`==` is not an assignment");
        // The keyword must be a whole word.
        assert!(
            binding_initializers("myconst notAKeyword = \"p-1\";").is_empty(),
            "`myconst` is not a declaration keyword",
        );
    }

    /// Regression: a binding declared INSIDE another binding's initializer — the
    /// `cva("base", { variants: { color: { yellow: "bg-yellow-500" } } })` shape, arrow
    /// bodies, IIFEs — must still be indexed. Resuming the scan past the enclosing
    /// initializer silently dropped every class string held in one (cal.com's
    /// `bg-yellow-500` / `text-yellow-500` / `text-rose-600` vanished from the emitted
    /// stylesheet with no error at all).
    #[test]
    fn binding_index_sees_declarations_nested_inside_another_initializer() {
        let source = r#"
            const outer = (() => {
              const nested = "bg-yellow-500";
              return nested;
            })();
            const after = "text-rose-600";
        "#;
        let index = binding_initializers(source);
        let names = index.iter().map(|(n, _)| *n).collect::<Vec<_>>();
        assert!(
            names.contains(&"outer"),
            "the outer binding is indexed: {names:?}"
        );
        assert!(
            names.contains(&"nested"),
            "the NESTED binding is indexed: {names:?}"
        );
        assert!(
            names.contains(&"after"),
            "scanning continues past the nesting: {names:?}"
        );

        // End to end: the class string only reachable through the nested binding
        // reaches the candidate set.
        let component = r#"
            const styles = (() => {
              const palette = { yellow: "bg-yellow-500", rose: "text-rose-600" };
              return palette;
            })();
            export const Badge = () => <span className={styles[tone]} />;
        "#;
        let mut out = BTreeSet::new();
        scan_class_candidates(component, &mut out);
        assert!(
            out.contains("bg-yellow-500"),
            "nested map value is a candidate: {out:?}"
        );
        assert!(
            out.contains("text-rose-600"),
            "nested map value is a candidate: {out:?}"
        );
    }

    /// The index is built across the whole source set, so a name bound in one file and
    /// referenced from another still resolves — and a name bound in SEVERAL files
    /// contributes every one of its initializers, not just the first.
    #[test]
    fn binding_index_spans_files_and_keeps_every_initializer_of_a_shared_name() {
        let light = r#"const TONE = "bg-white text-black";"#;
        let dark = r#"const TONE = "bg-black text-white";"#;
        let user = r#"export const C = () => <i className={TONE} />;"#;
        let mut out = BTreeSet::new();
        scan_class_candidates_multi(&[light, dark, user], &mut out);
        assert!(
            out.contains("bg-white"),
            "the first file's binding: {out:?}"
        );
        assert!(
            out.contains("bg-black"),
            "the second file's binding too: {out:?}"
        );
        assert!(out.contains("text-white"));
        assert!(out.contains("text-black"));
    }

    #[test]
    fn scans_safelist_arrays() {
        let mut out = BTreeSet::new();
        scan_class_candidates(
            r#"
            export default {
              content: ['./src/**/*.{ts,tsx}'],
              safelist: ['grid-cols-7', 'grid-rows-7'],
            }
            "#,
            &mut out,
        );
        assert!(out.contains("grid-cols-7"));
        assert!(out.contains("grid-rows-7"));
    }

    /// REGRESSION. A class-composition helper's arguments are class positions even
    /// when the call reaches `className` through a path no scan follows: cal.com's
    /// embed button reassigns a destructured PARAMETER
    /// (`className = classNames("hidden lg:inline-flex", className)`), so the JSX
    /// attribute holds a bare identifier and the binding index has no declaration
    /// to resolve it against. `lg:inline-flex` was the single utility the reference
    /// Tailwind build emitted that this one did not, and the button was `hidden` at
    /// every viewport (36px missing from the event-type header, caught by an
    /// element-by-element geometry diff against `next start`).
    #[test]
    fn scans_class_helper_call_arguments_anywhere_in_the_file() {
        let mut out = BTreeSet::new();
        scan_class_candidates(
            r#"
            export const EmbedButton = ({ className = "", ...props }) => {
              className = classNames("hidden lg:inline-flex", className);
              return <Component {...props} className={className} />;
            };
            function other({ style }) {
              const s = cn('flex-1', style && 'ring-2');
              const t = twMerge(`px-2 ${style}`, cva({ variants: { size: { sm: "text-xs" } } }));
              return s + t;
            }
            "#,
            &mut out,
        );
        assert!(out.contains("lg:inline-flex"), "{out:?}");
        assert!(out.contains("hidden"), "{out:?}");
        assert!(out.contains("flex-1"), "{out:?}");
        assert!(out.contains("ring-2"), "{out:?}");
        assert!(out.contains("px-2"), "{out:?}");
        assert!(out.contains("text-xs"), "{out:?}");
    }

    /// The helper name must be a WHOLE identifier: a call to something that merely
    /// ends in one of the names contributes nothing.
    #[test]
    fn a_name_ending_in_a_helper_name_is_not_a_class_helper() {
        let mut out = BTreeSet::new();
        scan_class_candidates(r#"const x = myCn("not-a-class-source");"#, &mut out);
        assert!(!out.contains("not-a-class-source"), "{out:?}");
    }

    #[test]
    fn math_operators_get_spaced_inside_math_functions() {
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&[
                "w-[min(800px,100dvh-280px)]",
                "max-w-[calc(100dvw-32px)]",
                "w-[calc(var(--x)-2px)]",
                "top-[-10px]",
            ]),
        )
        .unwrap();
        assert!(out.contains("{width:min(800px,100dvh - 280px)}"));
        assert!(out.contains("{max-width:calc(100dvw - 32px)}"));
        // `var(--x)` args stay untouched; the `-` after the call is spaced.
        assert!(out.contains("{width:calc(var(--x) - 2px)}"));
        // A leading sign is not an operator.
        assert!(out.contains("{top:-10px}"));
    }

    #[test]
    fn theme_calc_divisions_fold_like_the_reference_minifier() {
        // esbuild folds a/b only when multiplying by the reciprocal is
        // lossless in f64.
        assert_eq!(fold_exact_division("calc(1.5 / 1)").as_deref(), Some("1.5"));
        assert_eq!(
            fold_exact_division("calc(2.25 / 1.875)").as_deref(),
            Some("1.2")
        );
        assert_eq!(fold_exact_division("calc(1.75 / 1.25)"), None);
        assert_eq!(fold_exact_division("calc(1.25 / 0.875)"), None);
        assert_eq!(fold_exact_division("calc(2 / 1.5)"), None);
        let out = compile(
            "@import 'tailwindcss';",
            &candidates(&["text-base", "text-xl"]),
        )
        .unwrap();
        assert!(out.contains("--text-base--line-height:1.5;"));
        assert!(out.contains("--text-xl--line-height:calc(1.75 / 1.25)"));
    }

    #[test]
    fn app_installed_theme_overrides_the_vendored_defaults() {
        // The app's node_modules/tailwindcss/theme.css wins (default tokens
        // changed between v4 releases).
        let app_theme = "@theme default {\n  --font-sans: -apple-system, sans-serif;\n  --color-white: #fff;\n  --spacing: 0.25rem;\n}\n";
        let out = compile_with_theme(
            "@import 'tailwindcss';",
            &candidates(&["font-sans"]),
            Some(app_theme),
        )
        .unwrap();
        assert!(out.contains("--font-sans:-apple-system, sans-serif"));
        assert!(!out.contains("ui-sans-serif"));
    }

    #[test]
    fn preflight_and_banner_are_present_no_import_survives() {
        let out = compile(
            "@import 'tailwindcss' source('../');",
            &candidates(&["flex"]),
        )
        .unwrap();
        assert!(out.starts_with("/*! tailwindcss v4.3.3"));
        assert!(out.contains("@layer base{"));
        assert!(out.contains("box-sizing:border-box"));
        assert!(!out.contains("tailwindcss'"));
        assert!(!out.to_lowercase().contains("@import"));
    }

    // --- legacy Tailwind v3 dialect -----------------------------------------
    //
    // FINDINGS #19. A v3 app (`@tailwind base|components|utilities` + a
    // `tailwind.config.*`) compiled entirely against v4 reference data, so every one
    // of its design tokens and its whole base reset came out v4-shaped: colours in
    // `oklch()` where its own build emits `rgb()`, `rounded-full` at
    // `calc(infinity * 1px)` instead of `9999px`, and every element's default
    // border recoloured from gray-200 to `currentColor`. next-blog-starter showed
    // 194 computed-style differences across 60 elements.

    /// A resolved v3 config, in the `@theme` form `scripts/tailwind-config-eval.mjs`
    /// emits for it (see that script's `resolveConfig` path). Only the tokens these
    /// tests read.
    const V3_CONFIG_THEME: &str = "@theme{\
        --color-slate-400:#94a3b8;\
        --radius-full:9999px;\
        --default-border-color:#e5e7eb;\
    }";

    #[test]
    fn v3_entry_uses_the_v3_border_reset_not_v4s_currentcolor() {
        let v3 = compile_with_theme(
            "@tailwind base;\n@tailwind components;\n@tailwind utilities;\n",
            &candidates(&["flex"]),
            Some(&format!("{}\n{V3_CONFIG_THEME}", vendored_theme_css())),
        )
        .unwrap();
        // v3's preflight resets every border to `theme('borderColor.DEFAULT')`.
        assert!(v3.contains("border-color:#e5e7eb"), "{v3}");
        // ...and it is the v3 reset, not v4's (v4 has no `-webkit-tap-highlight-color`
        // reset in the same rule and does not ship v3's `:disabled{cursor:default}`).
        assert!(v3.contains(":disabled{cursor:default}"), "{v3}");

        // A v4 entry is untouched: it keeps v4's `currentColor` border reset.
        let v4 = compile("@import 'tailwindcss';", &candidates(&["flex"])).unwrap();
        assert!(!v4.contains("border-color:#e5e7eb"), "{v4}");
        assert!(!v4.contains(":disabled{cursor:default}"), "{v4}");
    }

    #[test]
    fn v3_theme_tokens_win_over_v4_built_in_literals() {
        let out = compile_with_theme(
            "@tailwind base;\n@tailwind utilities;\n",
            &candidates(&["rounded-full", "text-slate-400"]),
            Some(&format!("{}\n{V3_CONFIG_THEME}", vendored_theme_css())),
        )
        .unwrap();
        // v4 hard-codes `rounded-full` as `calc(infinity * 1px)`; a v3 config resolves
        // it to a real `--radius-full` token, which must win.
        assert!(
            out.contains(".rounded-full{border-radius:var(--radius-full)}"),
            "{out}"
        );
        assert!(out.contains("--radius-full:9999px"), "{out}");
        assert!(!out.contains("infinity"), "{out}");
        // The v3 palette (sRGB hex), not v4's oklch.
        assert!(out.contains("--color-slate-400:#94a3b8"), "{out}");
        assert!(!out.contains("oklch(0.704 0.04 256.788)"), "{out}");

        // With no v3 theme token, v4's own literal still applies.
        let v4 = compile("@import 'tailwindcss';", &candidates(&["rounded-full"])).unwrap();
        assert!(
            v4.contains(".rounded-full{border-radius:calc(infinity * 1px)}"),
            "{v4}"
        );
    }

    #[test]
    fn v3_preflight_theme_calls_all_resolve() {
        // Every `theme(...)` call in the vendored v3 preflight maps to a variable, and
        // none survives into the emitted base layer.
        let theme = Theme::parse(&format!("{}\n{V3_CONFIG_THEME}", vendored_theme_css()));
        let preflight = v3_preflight(&theme).unwrap();
        assert!(!preflight.contains("theme("), "{preflight}");
        // Resolved from the theme where a token exists...
        assert!(preflight.contains("border-color:#e5e7eb"), "{preflight}");
        // ...and from upstream's own literal fallback where it does not
        // (`colors.gray.400` is absent from V3_CONFIG_THEME, but present in the
        // vendored base, so the placeholder colour still resolves to a real value).
        assert!(preflight.contains("input::placeholder"), "{preflight}");
    }

    #[test]
    fn v3_theme_path_var_maps_the_paths_the_preflight_uses_and_rejects_others() {
        assert_eq!(
            v3_theme_path_var("borderColor.DEFAULT").as_deref(),
            Some("--default-border-color")
        );
        assert_eq!(
            v3_theme_path_var("colors.gray.400").as_deref(),
            Some("--color-gray-400")
        );
        assert_eq!(
            v3_theme_path_var("fontFamily.sans").as_deref(),
            Some("--font-sans")
        );
        assert_eq!(
            v3_theme_path_var("fontFamily.mono[1].fontFeatureSettings").as_deref(),
            Some("--font-mono--font-feature-settings")
        );
        // Unmapped paths are `None` so `v3_preflight` can raise a hard, named error
        // rather than emit a literal `theme(...)` the browser cannot parse.
        assert_eq!(v3_theme_path_var("spacing.4"), None);
        assert_eq!(v3_theme_path_var("colors"), None);
    }

    /// Compiles a legacy v3 entry against the vendored base plus `config_theme`
    /// (what `scripts/tailwind-config-eval.mjs` prints for the app's config).
    fn compile_v3(classes: &[&str], config_theme: &str) -> String {
        compile_with_theme(
            "@tailwind base;\n@tailwind components;\n@tailwind utilities;\n",
            &candidates(classes),
            Some(&format!("{}\n{config_theme}", vendored_theme_css())),
        )
        .unwrap()
    }

    #[test]
    fn v3_entry_box_shadow_composes_three_slots_not_v4s_five() {
        // next-blog-starter's cover images computed a 5-layer `box-shadow` where its
        // own tailwindcss 3.4.19 build computes 3: v4 added the `inset-shadow` and
        // `inset-ring` slots, and `shadow-sm`/`ring-*` assign the whole chain.
        let v3 = compile_v3(
            &["shadow-sm", "ring-2"],
            "@theme{--shadow-sm:0 5px 10px rgba(0,0,0,0.12);}",
        );
        assert!(
            v3.contains("box-shadow:var(--tw-ring-offset-shadow, 0 0 #0000), var(--tw-ring-shadow, 0 0 #0000), var(--tw-shadow)"),
            "{v3}"
        );
        assert!(!v3.contains("var(--tw-inset-ring-shadow)"), "{v3}");

        // A v4 entry keeps the 5-slot chain.
        let v4 = compile("@import 'tailwindcss';", &candidates(&["shadow-sm"])).unwrap();
        assert!(v4.contains("box-shadow:var(--tw-inset-shadow), var(--tw-inset-ring-shadow), var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow)"), "{v4}");
    }

    #[test]
    fn v3_entry_line_height_follows_source_order_not_v4s_tw_leading() {
        // v4 routes a `text-<size>`'s line-height through `var(--tw-leading, …)` so a
        // `leading-*` wins wherever it sits. v3 has no such slot: the two are plain
        // `line-height` declarations and the cascade decides. Compiling the v3 app the
        // v4 way made `md:text-4xl leading-tight` compute 45px where v3 computes 40px
        // (the `md:` group is emitted after every unprefixed utility, so it wins).
        let v3 = compile_v3(
            &[
                "text-3xl",
                "text-4xl",
                "md:text-4xl",
                "leading-tight",
                "leading-snug",
            ],
            "",
        );
        assert!(
            v3.contains(
                ".text-4xl{font-size:var(--text-4xl);line-height:var(--text-4xl--line-height)}"
            ),
            "{v3}"
        );
        assert!(
            v3.contains(".leading-tight{line-height:var(--leading-tight)}"),
            "{v3}"
        );
        assert!(!v3.contains("--tw-leading"), "{v3}");
        // v3's plugin order puts fontSize before lineHeight, so an unprefixed
        // `leading-*` still wins over an unprefixed `text-<size>`...
        let size_at = v3.find(".text-3xl{").unwrap();
        let leading_at = v3.find(".leading-snug{").unwrap();
        assert!(
            size_at < leading_at,
            "text-<size> must precede leading-*: {v3}"
        );
        // ...while a `md:` size, emitted in a later media group, beats it.
        assert!(v3.find(".md\\:text-4xl{").unwrap() > leading_at, "{v3}");

        let v4 = compile(
            "@import 'tailwindcss';",
            &candidates(&["text-4xl", "leading-tight"]),
        )
        .unwrap();
        assert!(
            v4.contains("line-height:var(--tw-leading, var(--text-4xl--line-height))"),
            "{v4}"
        );
    }

    #[test]
    fn v3_font_size_without_a_line_height_emits_only_font_size() {
        // `fontSize: { '5xl': '2.5rem' }` (a bare string) replaces v3's whole entry,
        // so v3 emits `.text-5xl{font-size:2.5rem}` with NO line-height. The vendored
        // v4 scale is cleared by the config's `--text-*: initial;` so its
        // `--text-5xl--line-height: 1` cannot leak back in.
        let v3 = compile_v3(
            &["text-5xl", "text-2xl"],
            "@theme{--text-*: initial;--text-2xl:1.5rem;--text-2xl--line-height:2rem;--text-5xl:2.5rem;}",
        );
        assert!(v3.contains(".text-5xl{font-size:var(--text-5xl)}"), "{v3}");
        assert!(
            v3.contains(
                ".text-2xl{font-size:var(--text-2xl);line-height:var(--text-2xl--line-height)}"
            ),
            "{v3}"
        );
        assert!(!v3.contains("--text-5xl--line-height"), "{v3}");
    }

    /// The v3 colours next-radix-ui's own config resolves to (sRGB hex, not v4 oklch).
    const V3_GRADIENT_THEME: &str =
        "@theme{--color-cyan-500:#06b6d4;--color-blue-500:#3b82f6;--color-red-500:#ef4444;}";

    #[test]
    fn v3_entry_gradients_interpolate_in_srgb_with_no_stop_positions() {
        // FINDINGS: next-radix-ui's hero computed
        // `linear-gradient(to right in oklab, rgb(6,182,212) 0%, rgb(59,130,246) 100%)`
        // where its own tailwindcss 3.4.9 build computes
        // `linear-gradient(to right, rgb(6,182,212), rgb(59,130,246))` — v4's oklab
        // interpolation and its `0%`/`100%` stop-position defaults, both absent in v3.
        let v3 = compile_v3(
            &[
                "bg-gradient-to-r",
                "from-cyan-500",
                "via-red-500",
                "to-blue-500",
                "from-transparent",
                "from-10%",
            ],
            V3_GRADIENT_THEME,
        );
        // The direction is written straight into `background-image`, in sRGB.
        assert!(
            v3.contains(".bg-gradient-to-r{background-image:linear-gradient(to right, var(--tw-gradient-stops))}"),
            "{v3}"
        );
        assert!(!v3.contains("in oklab"), "{v3}");
        assert!(!v3.contains("--tw-gradient-position"), "{v3}");
        // Every stop carries its own position var, and `from-*` fades `--tw-gradient-to`
        // to the same colour at zero alpha.
        assert!(
            v3.contains(".from-cyan-500{--tw-gradient-from:var(--color-cyan-500) var(--tw-gradient-from-position);--tw-gradient-to:rgb(6 182 212 / 0) var(--tw-gradient-to-position);--tw-gradient-stops:var(--tw-gradient-from), var(--tw-gradient-to)}"),
            "{v3}"
        );
        // `transparent` fades to `rgb(0 0 0 / 0)`, exactly as v3's `transparentTo` does.
        assert!(
            v3.contains(".from-transparent{--tw-gradient-from:transparent var(--tw-gradient-from-position);--tw-gradient-to:rgb(0 0 0 / 0) var(--tw-gradient-to-position);"),
            "{v3}"
        );
        // v3 has no `--tw-gradient-via`/`--tw-gradient-via-stops`: `via-*` inlines its
        // colour into the stop list.
        assert!(
            v3.contains(".via-red-500{--tw-gradient-to:rgb(239 68 68 / 0) var(--tw-gradient-to-position);--tw-gradient-stops:var(--tw-gradient-from), var(--color-red-500) var(--tw-gradient-via-position), var(--tw-gradient-to)}"),
            "{v3}"
        );
        assert!(!v3.contains("--tw-gradient-via-stops"), "{v3}");
        assert!(!v3.contains("--tw-gradient-via:"), "{v3}");
        // `to-*` sets only `--tw-gradient-to`, never the stop list.
        assert!(
            v3.contains(".to-blue-500{--tw-gradient-to:var(--color-blue-500) var(--tw-gradient-to-position)}"),
            "{v3}"
        );
        // The stop positions default to NOTHING — a plain rule, never `@property`
        // (a registered `syntax:"*"` property with no initial value is
        // guaranteed-invalid and would poison every stop declaration).
        assert!(
            v3.contains("*,::before,::after,::backdrop{--tw-gradient-from-position: ;--tw-gradient-via-position: ;--tw-gradient-to-position: }"),
            "{v3}"
        );
        assert!(!v3.contains("@property --tw-gradient"), "{v3}");
        // An explicit position utility still writes a real value.
        assert!(
            v3.contains(".from-10\\%{--tw-gradient-from-position:10%}"),
            "{v3}"
        );

        // A v4 entry keeps v4's composition untouched.
        let v4 = compile(
            "@import 'tailwindcss';",
            &candidates(&["bg-linear-to-r", "from-cyan-500", "to-blue-500"]),
        )
        .unwrap();
        assert!(
            v4.contains(".bg-linear-to-r{--tw-gradient-position:to right in oklab;"),
            "{v4}"
        );
        assert!(
            v4.contains(".from-cyan-500{--tw-gradient-from:var(--color-cyan-500);"),
            "{v4}"
        );
        assert!(
            v4.contains("@property --tw-gradient-from-position{syntax:\"<length-percentage>\";inherits:false;initial-value:0%}"),
            "{v4}"
        );
    }

    #[test]
    fn v3_entry_space_utilities_style_the_leading_edge_of_every_child_but_the_first() {
        // FINDINGS: next-radix-ui's `space-y-4` column computed `margin-top:0px;
        // margin-bottom:16px` on each child where its own build computes
        // `margin-top:16px; margin-bottom:0px`. v4 selects every child but the LAST
        // and pushes on the trailing edge; v3 selects every child but the FIRST and
        // pushes on the leading one.
        let v3 = compile_v3(
            &["space-y-4", "space-x-2", "space-y-0", "space-y-reverse"],
            "",
        );
        assert!(
            v3.contains(".space-y-4 > :not([hidden]) ~ :not([hidden]){--tw-space-y-reverse:0;margin-top:calc(calc(var(--spacing) * 4) * calc(1 - var(--tw-space-y-reverse)));margin-bottom:calc(calc(var(--spacing) * 4) * var(--tw-space-y-reverse))}"),
            "{v3}"
        );
        assert!(
            v3.contains(".space-x-2 > :not([hidden]) ~ :not([hidden]){--tw-space-x-reverse:0;margin-right:calc(calc(var(--spacing) * 2) * var(--tw-space-x-reverse));margin-left:calc(calc(var(--spacing) * 2) * calc(1 - var(--tw-space-x-reverse)))}"),
            "{v3}"
        );
        // v3 normalizes a `0` value to `0px` and keeps the calc; v4 folds it to `0`.
        assert!(
            v3.contains(".space-y-0 > :not([hidden]) ~ :not([hidden]){--tw-space-y-reverse:0;margin-top:calc(0px * calc(1 - var(--tw-space-y-reverse)));margin-bottom:calc(0px * var(--tw-space-y-reverse))}"),
            "{v3}"
        );
        assert!(
            v3.contains(
                ".space-y-reverse > :not([hidden]) ~ :not([hidden]){--tw-space-y-reverse:1}"
            ),
            "{v3}"
        );
        assert!(!v3.contains("margin-block-start"), "{v3}");
        assert!(!v3.contains(":not(:last-child)"), "{v3}");

        let v4 = compile("@import 'tailwindcss';", &candidates(&["space-y-4"])).unwrap();
        assert!(
            v4.contains(":where(.space-y-4 > :not(:last-child)){--tw-space-y-reverse:0;margin-block-start:calc(calc(var(--spacing) * 4) * var(--tw-space-y-reverse));margin-block-end:calc(calc(var(--spacing) * 4) * calc(1 - var(--tw-space-y-reverse)))}"),
            "{v4}"
        );
    }

    #[test]
    fn v3_entry_divide_utilities_share_the_v3_child_selector_and_edges() {
        // `divide-*` rides the same between-children selector as `space-*`, so the
        // v3 selector must come with v3's edges: flipping one without the other
        // would draw every rule on the wrong side of the wrong child.
        let v3 = compile_v3(
            &["divide-y-2", "divide-x", "divide-x-reverse", "divide-solid"],
            "",
        );
        assert!(
            v3.contains(".divide-y-2 > :not([hidden]) ~ :not([hidden]){--tw-divide-y-reverse:0;border-top-width:calc(2px * calc(1 - var(--tw-divide-y-reverse)));border-bottom-width:calc(2px * var(--tw-divide-y-reverse))}"),
            "{v3}"
        );
        assert!(
            v3.contains(".divide-x > :not([hidden]) ~ :not([hidden]){--tw-divide-x-reverse:0;border-right-width:calc(1px * var(--tw-divide-x-reverse));border-left-width:calc(1px * calc(1 - var(--tw-divide-x-reverse)))}"),
            "{v3}"
        );
        assert!(
            v3.contains(
                ".divide-x-reverse > :not([hidden]) ~ :not([hidden]){--tw-divide-x-reverse:1}"
            ),
            "{v3}"
        );
        // v3 has no `--tw-border-style` slot; the width rules lean on the preflight's
        // `border-style: solid` and `divide-<style>` writes the real property alone.
        assert!(
            v3.contains(".divide-solid > :not([hidden]) ~ :not([hidden]){border-style:solid}"),
            "{v3}"
        );
        assert!(!v3.contains("--tw-border-style"), "{v3}");
        assert!(!v3.contains("border-inline-start-width"), "{v3}");

        let v4 = compile("@import 'tailwindcss';", &candidates(&["divide-y-2"])).unwrap();
        assert!(
            v4.contains(":where(.divide-y-2 > :not(:last-child)){--tw-divide-y-reverse:0;border-bottom-style:var(--tw-border-style);border-top-style:var(--tw-border-style);border-top-width:calc(2px * var(--tw-divide-y-reverse));border-bottom-width:calc(2px * calc(1 - var(--tw-divide-y-reverse)))}"),
            "{v4}"
        );
    }

    #[test]
    fn v3_transparent_color_matches_upstreams_transparent_to() {
        // Upstream v3's `transparentTo(value)` = `withAlphaValue(value, 0, 'rgb(255 255 255 / 0)')`.
        assert_eq!(v3_transparent_color(Some("#06b6d4")), "rgb(6 182 212 / 0)");
        assert_eq!(v3_transparent_color(Some("#abc")), "rgb(170 187 204 / 0)");
        assert_eq!(v3_transparent_color(Some("#000")), "rgb(0 0 0 / 0)");
        assert_eq!(v3_transparent_color(Some("rgb(1,2,3)")), "rgb(1 2 3 / 0)");
        assert_eq!(
            v3_transparent_color(Some("rgba(1 2 3 / 0.5)")),
            "rgb(1 2 3 / 0)"
        );
        assert_eq!(v3_transparent_color(Some("transparent")), "rgb(0 0 0 / 0)");
        // Anything v3's parser cannot read falls back to transparent white.
        assert_eq!(
            v3_transparent_color(Some("currentcolor")),
            "rgb(255 255 255 / 0)"
        );
        assert_eq!(
            v3_transparent_color(Some("inherit")),
            "rgb(255 255 255 / 0)"
        );
        assert_eq!(
            v3_transparent_color(Some("var(--x)")),
            "rgb(255 255 255 / 0)"
        );
        assert_eq!(v3_transparent_color(None), "rgb(255 255 255 / 0)");
    }

    #[test]
    fn theme_namespace_wildcard_clears_earlier_tokens_and_rejects_other_values() {
        let theme = Theme::parse(
            "--text-sm:1rem;--text-sm--line-height:2;--color-red:#f00;--text-*: initial;--text-sm:0.5rem;",
        );
        assert_eq!(theme.get("--text-sm"), Some("0.5rem"));
        assert_eq!(theme.get("--text-sm--line-height"), None);
        // A different namespace is untouched.
        assert_eq!(theme.get("--color-red"), Some("#f00"));
        assert_eq!(
            theme.order,
            vec!["--color-red".to_string(), "--text-sm".to_string()]
        );

        Theme::validate_wildcards("--text-*: initial;").unwrap();
        let error = Theme::validate_wildcards("--text-*: 1rem;").unwrap_err();
        assert!(error.contains("--text-*"), "{error}");
        assert!(error.contains("initial"), "{error}");
    }

    #[test]
    fn v3_config_dark_mode_class_makes_dark_a_selector_variant() {
        // `darkMode: "class"` reaches the compiler as a `@custom-variant dark (…)`
        // in the config-derived theme source. Dropping it compiled every `dark:`
        // utility into `@media (prefers-color-scheme: dark)`, so next-blog-starter
        // painted its dark palette on a browser that merely preferred dark.
        let v3 = compile_v3(
            &["dark:text-slate-400"],
            "@custom-variant dark (&:is(.dark *));\n@theme{--color-slate-400:#94a3b8;}",
        );
        assert!(
            v3.contains(".dark\\:text-slate-400:is(.dark *){color:var(--color-slate-400)}"),
            "{v3}"
        );
        assert!(!v3.contains("prefers-color-scheme"), "{v3}");

        // With no `darkMode` in the config (v3's default is `media`) the media query
        // is still what `dark:` means.
        let media = compile_v3(
            &["dark:text-slate-400"],
            "@theme{--color-slate-400:#94a3b8;}",
        );
        assert!(
            media.contains("@media (prefers-color-scheme: dark){"),
            "{media}"
        );
    }

    #[test]
    fn theme_source_custom_variants_are_scanned_and_the_app_css_overrides_them() {
        let found = scan_custom_variants(
            "@theme{--color-a:#000;}\n@custom-variant dark (&:is(.dark *));\n@custom-variant hocus (&:hover);",
        )
        .unwrap();
        assert_eq!(found.get("dark").map(String::as_str), Some("&:is(.dark *)"));
        assert_eq!(found.get("hocus").map(String::as_str), Some("&:hover"));
        assert!(scan_custom_variants("--color-a:#000;").unwrap().is_empty());

        // An app stylesheet that declares its own `dark` variant wins over the config's.
        let out = compile_with_theme(
            "@import 'tailwindcss';\n@custom-variant dark (&:where([data-theme=dark] *));",
            &candidates(&["dark:text-white"]),
            Some(&format!(
                "{}\n@custom-variant dark (&:is(.dark *));",
                vendored_theme_css()
            )),
        )
        .unwrap();
        assert!(out.contains(":where([data-theme=dark] *)"), "{out}");
        assert!(!out.contains(":is(.dark *)"), "{out}");
    }

    #[test]
    fn a_functional_utility_generates_one_rule_per_value_form_it_accepts() {
        // Tailwind v4's `@utility <prefix>-*`: the candidate's value is resolved by
        // the `--value(…)` calls in the body, a declaration whose call resolves to
        // nothing is DROPPED, and a candidate that drops every declaration
        // generates no rule at all. The two-`margin-top` shape below is the
        // idiomatic way to accept both a spacing step and an arbitrary length.
        let css = "@import 'tailwindcss';\n\
                   @utility stack-y-* {\n\
                     & > :not([hidden]) ~ :not([hidden]) {\n\
                       margin-top: --spacing(--value(integer));\n\
                       margin-top: --value([length]);\n\
                     }\n\
                   }\n";
        let out = compile(
            css,
            &candidates(&["stack-y-4", "stack-y-[3px]", "stack-y-px", "sm:stack-y-0"]),
        )
        .unwrap();

        // A bare integer takes the `--spacing()` branch.
        assert!(
            out.contains(
                ".stack-y-4{& > :not([hidden]) ~ :not([hidden]){margin-top:calc(var(--spacing) * 4);}}"
            ),
            "{out}"
        );
        // An arbitrary length takes the other branch, and only that one.
        assert!(
            out.contains(
                ".stack-y-\\[3px\\]{& > :not([hidden]) ~ :not([hidden]){margin-top:3px;}}"
            ),
            "{out}"
        );
        // `px` is neither an integer nor a bracketed length: nothing is generated,
        // exactly as Tailwind's own output for this definition.
        assert!(!out.contains("stack-y-px"), "{out}");
        // Variants compose with a custom utility like any other.
        assert!(
            out.contains("@media (width >= 40rem){.sm\\:stack-y-0{"),
            "{out}"
        );
        // The `--spacing` theme token it references is pulled into the theme layer.
        assert!(out.contains("--spacing:"), "{out}");
    }

    #[test]
    fn a_static_utility_and_a_theme_namespace_value_resolve_through_at_utility() {
        let css = "@import 'tailwindcss';\n\
                   @theme { --leading-cozy: 1.4; }\n\
                   @utility no-scrollbar { scrollbar-width: none; }\n\
                   @utility lead-* { line-height: --value(--leading-*); }\n\
                   .uses-apply { @apply no-scrollbar; }\n";
        let out = compile(
            css,
            &candidates(&[
                "no-scrollbar",
                "lead-cozy",
                "lead-nonexistent",
                "hover:no-scrollbar",
            ]),
        )
        .unwrap();

        assert!(
            out.contains(".no-scrollbar{scrollbar-width:none;}"),
            "{out}"
        );
        assert!(
            out.contains(".hover\\:no-scrollbar:hover{scrollbar-width:none;}"),
            "{out}"
        );
        // `--value('--leading-*')` is a theme lookup: it emits the token reference.
        assert!(
            out.contains(".lead-cozy{line-height:var(--leading-cozy);}"),
            "{out}"
        );
        assert!(out.contains("--leading-cozy:1.4"), "{out}");
        // A value that names no token generates nothing (Tailwind's behaviour).
        assert!(!out.contains("lead-nonexistent"), "{out}");
        // `@apply` of an app-defined utility flattens into the applying rule.
        assert!(out.contains(".uses-apply{scrollbar-width:none}"), "{out}");
    }

    #[test]
    fn an_unimplemented_value_function_or_data_type_is_a_hard_error() {
        // Never a silent non-match: a construct the engine does not implement must
        // name itself, or an app ships with the style quietly missing.
        let unknown_fn = compile(
            "@import 'tailwindcss';\n@utility tint-* { color: --alpha(--value(color) / 50%); }\n",
            &candidates(&["tint-red"]),
        )
        .unwrap_err();
        assert!(unknown_fn.contains("--alpha()"), "{unknown_fn}");

        let unknown_type = compile(
            "@import 'tailwindcss';\n@utility grid-area-* { grid-area: --value(position); }\n",
            &candidates(&["grid-area-main"]),
        )
        .unwrap_err();
        assert!(unknown_type.contains("position"), "{unknown_type}");
    }

    #[test]
    fn a_longer_functional_prefix_wins_and_an_app_utility_overrides_the_builtin() {
        let css = "@import 'tailwindcss';\n\
                   @utility stack-* { padding: --spacing(--value(integer)); }\n\
                   @utility stack-y-* { margin: --spacing(--value(integer)); }\n\
                   @utility p-* { outline-width: --value(integer); }\n";
        let out = compile(css, &candidates(&["stack-y-2", "stack-2", "p-3"])).unwrap();
        assert!(
            out.contains(".stack-y-2{margin:calc(var(--spacing) * 2);}"),
            "{out}"
        );
        assert!(
            out.contains(".stack-2{padding:calc(var(--spacing) * 2);}"),
            "{out}"
        );
        // The app's own `p-*` replaces Tailwind's padding utility entirely.
        assert!(out.contains(".p-3{outline-width:3;}"), "{out}");
        assert!(!out.contains(".p-3{padding"), "{out}");
    }

    #[test]
    fn value_data_types_classify_css_values() {
        let check = |value: &str, ty: &str| match value_matches_data_type(value, ty, "t", "--value")
        {
            Ok(matched) => matched,
            Err(Fail::Unsupported(error)) => panic!("{error}"),
            Err(Fail::Invalid) => panic!("unexpected Invalid for {value:?} as {ty}"),
        };
        assert!(check("4", "integer"));
        assert!(check("-4", "integer"));
        assert!(!check("4.5", "integer"));
        assert!(check("4.5", "number"));
        assert!(check("3px", "length"));
        assert!(check("0", "length"));
        assert!(!check("3", "length"));
        assert!(check("calc(100% - 3px)", "length"));
        assert!(check("50%", "percentage"));
        assert!(check("16/9", "ratio"));
        assert!(check("45deg", "angle"));
        assert!(check("150ms", "time"));
        assert!(check("#abc", "color"));
        assert!(check("oklch(0.5 0.1 20)", "color"));
        assert!(check("currentColor", "color"));
        assert!(!check("3px", "color"));
        assert!(check("anything at all", "*"));
    }

    // -----------------------------------------------------------------------
    // Native-engine capability check / delegation gate
    // -----------------------------------------------------------------------

    /// The whole performance story: an app using only what diffpack implements
    /// must stay on the native path, with no `node` spawn. Every shape the native
    /// engine owns has to answer "no gap".
    #[test]
    fn a_sheet_the_native_engine_owns_reports_no_gap() {
        let css = "@import 'tailwindcss';\n\
                   @theme inline { --color-brand: #123456; }\n\
                   @custom-variant dark (&:where(.dark, .dark *));\n\
                   @utility tab-4 { tab-size: 4; }\n\
                   @source '../src';\n\
                   @layer base { html, body { @apply bg-brand text-white; } }\n\
                   @layer components { .card { @apply rounded-lg p-4 tab-4; } }\n\
                   @keyframes spin { to { transform: rotate(360deg); } }\n\
                   @media (min-width: 40rem) { .x { color: red; } }\n\
                   .plain { color: blue; }\n";
        assert_eq!(native_gap(css, None), None);
    }

    /// The four Tailwind-bearing corpus apps must never start spawning node: their
    /// entries are the byte-identity guard for the native fast path.
    #[test]
    fn every_tailwind_corpus_entry_stays_native() {
        let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let entries = [
            "integration/e2e/apps/tanstack-start-basic/src/styles/app.css",
            "integration/e2e/apps/tanstack-start-tailwind-v4/src/styles/app.css",
            "integration/e2e/apps/next-blog-starter/src/app/globals.css",
            "integration/e2e/apps/next-radix-ui/styles/globals.css",
        ];
        let mut checked = 0;
        for relative in entries {
            let path = repo.join(relative);
            let Ok(css) = std::fs::read_to_string(&path) else {
                // The corpus apps are fetched on demand; skip what is not present.
                continue;
            };
            checked += 1;
            assert_eq!(
                native_gap(&css, None),
                None,
                "{relative} must compile natively (no node spawn, byte-identical sheet)"
            );
        }
        if checked == 0 {
            eprintln!("skipping Tailwind corpus guard: external corpus apps are not installed");
        }
    }

    #[test]
    fn a_javascript_plugin_is_a_native_gap() {
        let gap = native_gap(
            "@import 'tailwindcss';\n@plugin 'tailwind-scrollbar' {\n  nocompatible: true;\n}\n",
            None,
        );
        assert_eq!(
            gap,
            Some(NativeGap::Plugin("tailwind-scrollbar".to_string()))
        );
        assert!(gap.unwrap().to_string().contains("tailwind-scrollbar"));
    }

    #[test]
    fn an_at_rule_the_native_engine_has_no_meaning_for_is_a_gap() {
        // `@variant name (…)` at top level is Tailwind's newer spelling of
        // `@custom-variant`; the native engine only implements the latter.
        assert_eq!(
            native_gap(
                "@import 'tailwindcss';\n@variant pwa (@media (display-mode: standalone));\n",
                None
            ),
            Some(NativeGap::AtRule("@variant".to_string()))
        );
    }

    /// The at-rule scan is a real CSS scan, not a substring search: an at-rule
    /// inside a block, a string or a comment is not a top-level directive.
    #[test]
    fn at_rules_that_are_not_top_level_directives_are_not_gaps() {
        let css = "@import 'tailwindcss';\n\
                   /* @plugin 'daisyui'; */\n\
                   .a::after { content: '@plugin \\\"x\\\"'; }\n\
                   @media (min-width: 40rem) { .b { @apply flex; } }\n";
        assert_eq!(native_gap(css, None), None);
    }

    #[test]
    fn applying_a_utility_only_a_plugin_registers_is_a_gap() {
        // `scrollbar-thumb-rounded-md` comes from `tailwind-scrollbar`; nothing in
        // the theme or in an `@utility` can answer it.
        let css = "@import 'tailwindcss';\n\
                   @layer components { .scroll-bar { @apply scrollbar-thumb-rounded-md; } }\n";
        match native_gap(css, None) {
            Some(NativeGap::Apply { class, .. }) => {
                assert_eq!(class, "scrollbar-thumb-rounded-md");
            }
            other => panic!("expected an @apply gap, got {other:?}"),
        }
    }

    /// An `@apply` an app `@utility` answers is served natively — the check must
    /// not delegate a sheet the engine can compile.
    #[test]
    fn applying_an_app_defined_utility_is_not_a_gap() {
        let css = "@import 'tailwindcss';\n\
                   @utility scroll-thin { scrollbar-width: thin; }\n\
                   .scroll-bar { @apply scroll-thin; }\n";
        assert_eq!(native_gap(css, None), None);
    }

    /// A variant the native engine cannot expand in `@apply` is a capability gap
    /// like any other: Tailwind itself supports it, so the app's own compiler can.
    #[test]
    fn an_unexpandable_apply_variant_is_a_gap() {
        let css = "@import 'tailwindcss';\n.b { @apply hover:bg-black; }\n";
        assert!(matches!(
            native_gap(css, None),
            Some(NativeGap::Apply { .. })
        ));
    }

    /// Broken CSS is NOT a capability gap. Delegating it would turn diffpack's own
    /// diagnostic into a foreign one; it stays a native hard error.
    #[test]
    fn malformed_css_is_not_a_gap() {
        assert_eq!(
            native_gap("@import 'tailwindcss';\n.a { color: red;\n", None),
            None
        );
        assert_eq!(
            native_gap("@import 'tailwindcss';\n@utility { color: red; }\n", None),
            None
        );
    }

    /// `@apply` resolution is answered against the theme actually in scope: a token
    /// the app's installed Tailwind defines makes the utility native.
    #[test]
    fn the_apply_probe_uses_the_supplied_theme() {
        let css = "@import 'tailwindcss';\n.b { @apply bg-brandish; }\n";
        assert!(matches!(
            native_gap(css, None),
            Some(NativeGap::Apply { .. })
        ));
        let theme = format!("{THEME_CSS}\n@theme {{ --color-brandish: #abcdef; }}\n");
        assert_eq!(native_gap(css, Some(&theme)), None);
    }
}
