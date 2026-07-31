//! Native `next/font` support — the hard blocker on the stock create-next-app.
//!
//! `next/font/google` and `next/font/local` are **build-time macros**, not runtime
//! modules: the published npm package is a placeholder, and Next's SWC loader
//! REPLACES each `Geist({...})` / `localFont({...})` call with a generated object
//! (`{ className, variable, style }`) plus the font's CSS (`@font-face` / a CSS
//! variable class). Importing the real module and calling it throws. Diffpack does
//! the same rewrite natively on the oxc AST (source-to-source, gated on a cheap
//! `next/font` string check, so non-font modules pay nothing), and generates the
//! companion CSS ([`generate`]) which the app-router adapter injects into the
//! document `<head>`.
//!
//! Fidelity, `next/font/google`: the family's real webfont is loaded via a Google
//! Fonts `@import` (so the browser fetches the actual font), and the call's `variable`
//! option is wired to a CSS-variable class exactly as Next does, so `${font.variable}`
//! on `<html>` defines the custom property the app's CSS reads. Self-hosting the
//! google font files (Next's default, to avoid the external request) is a later
//! refinement.
//!
//! Fidelity, `next/font/local`: the font FILE is the source of truth, so diffpack
//! reads it. Each `src` entry is resolved relative to the calling module, emitted as a
//! content-hashed build asset under `/_diffpack-font/`, and given the `@font-face`
//! Next's own loader (`@next/font/dist/local/loader.js`) generates — same property
//! order, same `format()`, same `font-display`/`font-weight`/`font-style`, same
//! `declarations` passthrough. The family name is the name of the const the call is
//! assigned to, which is what Next uses (`variableName`); a call not assigned to one
//! is a hard error, exactly as it is under Next.
//!
//! The other half of what `next/font` IS — and the entire reason it exists — is the
//! metric-matched fallback face: Next emits a local `@font-face` (`Inter Fallback`,
//! `src: local("Arial")`) carrying `size-adjust` / `ascent-override` /
//! `descent-override` / `line-gap-override` computed from the family's real metrics,
//! so the page does not reflow when the webfont finishes loading. Diffpack computes
//! the same four numbers with Next's own arithmetic, from the same input Next uses on
//! each side:
//!
//! * google — the `capsize-font-metrics` table shipped inside the app's own `next`
//!   install (`next/dist/server/font-utils.js`);
//! * local — the font binary itself, read by [`crate::font_file`], which is what
//!   Next's fontkit pass does (`getFallbackMetricsFromFontFile`).
//!
//! Either way a family whose metrics cannot be derived is a HARD ERROR: silently
//! dropping the face is exactly the layout shift `next/font` is there to prevent.

use std::path::{Path, PathBuf};

use oxc_allocator::Allocator;
use oxc_ast::ast::{Argument, Expression, ObjectPropertyKind, PropertyKey, Statement};
use oxc_ast_visit::{walk, Visit};
use oxc_parser::Parser;
use oxc_span::Span;

use crate::server_fn::{apply_edits, quote};

/// Where the emitted local-font files are served from, relative to the site root —
/// the same shape as `next/image`'s `_diffpack-image/`. Written into the client
/// build's `public/` output, so the static-asset path serves them with no per-request
/// cost.
pub const FONT_ASSET_DIR: &str = "_diffpack-font";

/// The build-scoped record of which font files the emitted CSS references, written
/// under the Next adapter directory by [`generate`] and consumed by
/// [`emit_font_assets`]. It exists so the client emit can copy the font binaries
/// WITHOUT walking the project a second time: the same pass that produced the URLs
/// records the sources they came from, and the two therefore cannot disagree.
pub const FONT_MANIFEST_FILE: &str = "fonts.json";

/// A resolved `next/font` usage: the display family (`"Geist Mono"`, or for a local
/// font the name of the const it is assigned to), the CSS variable name from the
/// call's `variable` option (if any), whether it is a Google font, and the
/// deterministic class names the rewrite and the CSS agree on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FontUsage {
    pub family: String,
    pub variable: Option<String>,
    pub class_name: String,
    pub variable_class: String,
    pub google: bool,
    /// The call's `fallback: ["Helvetica", ...]` option, appended to the family list
    /// verbatim exactly as Next does.
    pub fallbacks: Vec<String>,
    /// Whether Next generates the metric-matched `"<family> Fallback"` face
    /// (`adjustFontFallback`, default on).
    pub adjust_fallback: bool,
    /// Everything a `next/font/local` usage needs that a google one does not: the font
    /// files, the calling module they resolve against, and the `@font-face` options.
    pub local: Option<LocalFont>,
}

/// One `src` entry of a `next/font/local` call, after Next's normalisation
/// (`validateLocalFontFunctionCall`): a bare `src: "./x.woff2"` becomes a single entry
/// carrying the call's top-level `weight`/`style`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalSrc {
    /// The specifier as written, resolved relative to the calling module.
    pub path: String,
    pub weight: Option<String>,
    pub style: Option<String>,
}

/// Which metric-matched fallback face a `next/font/local` call asks for.
/// `adjustFontFallback` is `false` | `'Arial'` | `'Times New Roman'` there (not the
/// boolean the google loader takes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalAdjust {
    /// The default (or an explicit `'Arial'`): scale Arial to match.
    SansSerif,
    /// `adjustFontFallback: 'Times New Roman'`.
    Serif,
    /// `adjustFontFallback: false` — the documented opt-out.
    Off,
}

/// A `next/font/local` call's emit-relevant state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalFont {
    /// The module the call sits in; every `src` path is relative to its directory.
    pub module: PathBuf,
    pub src: Vec<LocalSrc>,
    /// `display`, defaulting to `swap` as Next does.
    pub display: String,
    /// `preload` (default `true`): whether the head gets a `<link rel="preload">`.
    pub preload: bool,
    /// `declarations: [{ prop, value }]`, emitted verbatim ahead of the generated
    /// descriptors.
    pub declarations: Vec<(String, String)>,
    pub adjust: LocalAdjust,
    /// Next returns `weight`/`style` on the font OBJECT only when there is exactly one
    /// `src` entry (`weight: src.length === 1 ? src[0].weight : undefined`).
    pub weight: Option<String>,
    pub style: Option<String>,
}

/// What one `next/font` generation produced: the CSS to inject, the font files to copy
/// into the served `public/`, and the hrefs that want a `<link rel="preload">`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FontOutput {
    pub css: String,
    pub assets: Vec<FontAsset>,
    pub preloads: Vec<String>,
}

/// A local font file the build must copy into the served `public/` output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FontAsset {
    /// Absolute path of the source font file.
    pub source: PathBuf,
    /// Path under the served `public/` root (`_diffpack-font/Cal-1a2b3c4d.woff2`).
    pub file: String,
}

/// The deterministic class-name slug for a font binding (`Geist_Mono` -> `geist_mono`).
fn slug(binding: &str) -> String {
    binding
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
        .collect()
}

/// The Google family display name for an imported binding (`Geist_Mono` -> `Geist Mono`).
fn family_display(binding: &str) -> String {
    binding.replace('_', " ")
}

/// One imported font-factory binding: its local name and whether it comes from
/// `next/font/google` (vs `next/font/local`).
struct FontImport {
    local: String,
    google: bool,
}

/// Collects the `next/font/*` import bindings (named for google, default for local),
/// and the spans of the import declarations (to delete them from the rewrite).
fn collect_font_imports(program: &oxc_ast::ast::Program) -> (Vec<FontImport>, Vec<Span>) {
    let mut imports = Vec::new();
    let mut import_spans = Vec::new();
    for statement in &program.body {
        let Statement::ImportDeclaration(decl) = statement else { continue };
        let source = decl.source.value.as_str();
        let google = source == "next/font/google";
        if !google && source != "next/font/local" {
            continue;
        }
        import_spans.push(decl.span);
        let Some(specifiers) = &decl.specifiers else { continue };
        for specifier in specifiers {
            use oxc_ast::ast::ImportDeclarationSpecifier as Spec;
            match specifier {
                // `import { Geist, Geist_Mono } from "next/font/google"` — the
                // imported name IS the Google family.
                Spec::ImportSpecifier(spec) => {
                    imports.push(FontImport { local: spec.local.name.to_string(), google });
                }
                // `import localFont from "next/font/local"` — the family is the name of
                // the const the call is assigned to, resolved per call site.
                Spec::ImportDefaultSpecifier(spec) => {
                    imports.push(FontImport { local: spec.local.name.to_string(), google });
                }
                Spec::ImportNamespaceSpecifier(_) => {}
            }
        }
    }
    (imports, import_spans)
}

/// A matched call `Font({...})`: the whole call span (to replace) and the raw options.
struct MatchedCall {
    span: Span,
    options: RawOptions,
}

/// Visitor collecting every call whose callee is one of the font bindings, plus the
/// name of the `const` each call is assigned to — which is the family name for
/// `next/font/local` (Next's `variableName`) and the class-name slug for it.
struct CallCollector<'a> {
    names: &'a [String],
    calls: Vec<(String, MatchedCall)>,
    /// call span -> declarator name.
    declared: Vec<(Span, String)>,
}

impl<'a> Visit<'a> for CallCollector<'a> {
    fn visit_variable_declarator(&mut self, decl: &oxc_ast::ast::VariableDeclarator<'a>) {
        if let Some(Expression::CallExpression(call)) = &decl.init
            && let Expression::Identifier(ident) = &call.callee
            && self.names.iter().any(|n| n == ident.name.as_str())
            && let Some(id) = decl.id.get_binding_identifier()
        {
            self.declared.push((call.span, id.name.to_string()));
        }
        walk::walk_variable_declarator(self, decl);
    }

    fn visit_call_expression(&mut self, call: &oxc_ast::ast::CallExpression<'a>) {
        if let Expression::Identifier(ident) = &call.callee
            && self.names.iter().any(|n| n == ident.name.as_str())
        {
            let options = call.arguments.first().map(read_options).unwrap_or_default();
            self.calls.push((ident.name.to_string(), MatchedCall { span: call.span, options }));
        }
        walk::walk_call_expression(self, call);
    }
}

/// The literal option values read off a font call's argument object, before Next's
/// normalisation. Kept raw because normalisation depends on values that may appear
/// AFTER the one that consumes them (`src: "./x.woff2"` inherits the call's top-level
/// `weight`, whichever order the properties are written in).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct RawOptions {
    variable: Option<String>,
    fallbacks: Vec<String>,
    /// `adjustFontFallback` as written: `Some(false)` for the boolean opt-out,
    /// `Some(true)` for the boolean opt-in, and the string form for local fonts.
    adjust_bool: Option<bool>,
    adjust_string: Option<String>,
    /// `src`, either a bare string or an array of `{ path, weight, style }`.
    src_string: Option<String>,
    src_array: Option<Vec<LocalSrc>>,
    /// True when `src` was present but neither of the two accepted shapes.
    src_unsupported: bool,
    display: Option<String>,
    weight: Option<String>,
    style: Option<String>,
    preload: Option<bool>,
    declarations: Vec<(String, String)>,
}

/// Reads the emit-relevant options out of a font call's first argument.
fn read_options(arg: &Argument) -> RawOptions {
    let mut options = RawOptions::default();
    let Argument::ObjectExpression(object) = arg else { return options };
    for property in &object.properties {
        let ObjectPropertyKind::ObjectProperty(prop) = property else { continue };
        let key = match &prop.key {
            PropertyKey::StaticIdentifier(ident) => ident.name.as_str(),
            PropertyKey::StringLiteral(lit) => lit.value.as_str(),
            _ => continue,
        };
        match key {
            "variable" => options.variable = string_of(&prop.value),
            "fallback" => {
                if let Expression::ArrayExpression(array) = &prop.value {
                    for element in &array.elements {
                        if let Some(value) = element.as_expression().and_then(string_of) {
                            options.fallbacks.push(value);
                        }
                    }
                }
            }
            "adjustFontFallback" => match &prop.value {
                Expression::BooleanLiteral(value) => options.adjust_bool = Some(value.value),
                other => options.adjust_string = string_of(other),
            },
            "display" => options.display = string_of(&prop.value),
            "weight" => options.weight = string_or_number_of(&prop.value),
            "style" => options.style = string_of(&prop.value),
            "preload" => {
                if let Expression::BooleanLiteral(value) = &prop.value {
                    options.preload = Some(value.value);
                }
            }
            "src" => match &prop.value {
                Expression::StringLiteral(value) => {
                    options.src_string = Some(value.value.to_string())
                }
                Expression::ArrayExpression(array) => {
                    let mut entries = Vec::new();
                    for element in &array.elements {
                        let Some(Expression::ObjectExpression(entry)) = element.as_expression()
                        else {
                            options.src_unsupported = true;
                            continue;
                        };
                        let mut src =
                            LocalSrc { path: String::new(), weight: None, style: None };
                        for field in &entry.properties {
                            let ObjectPropertyKind::ObjectProperty(field) = field else { continue };
                            let name = match &field.key {
                                PropertyKey::StaticIdentifier(ident) => ident.name.as_str(),
                                PropertyKey::StringLiteral(lit) => lit.value.as_str(),
                                _ => continue,
                            };
                            match name {
                                "path" => {
                                    src.path = string_of(&field.value).unwrap_or_default();
                                }
                                "weight" => src.weight = string_or_number_of(&field.value),
                                "style" => src.style = string_of(&field.value),
                                _ => {}
                            }
                        }
                        if src.path.is_empty() {
                            options.src_unsupported = true;
                        } else {
                            entries.push(src);
                        }
                    }
                    options.src_array = Some(entries);
                }
                _ => options.src_unsupported = true,
            },
            "declarations" => {
                if let Expression::ArrayExpression(array) = &prop.value {
                    for element in &array.elements {
                        let Some(Expression::ObjectExpression(entry)) = element.as_expression()
                        else {
                            continue;
                        };
                        let mut declaration = (String::new(), String::new());
                        for field in &entry.properties {
                            let ObjectPropertyKind::ObjectProperty(field) = field else { continue };
                            let name = match &field.key {
                                PropertyKey::StaticIdentifier(ident) => ident.name.as_str(),
                                PropertyKey::StringLiteral(lit) => lit.value.as_str(),
                                _ => continue,
                            };
                            match name {
                                "prop" => declaration.0 = string_of(&field.value).unwrap_or_default(),
                                "value" => {
                                    declaration.1 =
                                        string_or_number_of(&field.value).unwrap_or_default()
                                }
                                _ => {}
                            }
                        }
                        if !declaration.0.is_empty() {
                            options.declarations.push(declaration);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    options
}

fn string_of(expression: &Expression) -> Option<String> {
    match expression {
        Expression::StringLiteral(value) => Some(value.value.to_string()),
        _ => None,
    }
}

/// `weight: 600` and `weight: "600"` mean the same thing to Next (the SWC transform
/// hands the loader whatever literal was written, and the CSS stringifies it).
fn string_or_number_of(expression: &Expression) -> Option<String> {
    match expression {
        Expression::StringLiteral(value) => Some(value.value.to_string()),
        Expression::NumericLiteral(value) => Some(value.raw.as_ref()?.to_string()),
        _ => None,
    }
}

/// `next/font`'s allowed `display` values (`@next/font/dist/constants.js`).
const ALLOWED_DISPLAY: [&str; 5] = ["auto", "block", "swap", "fallback", "optional"];

/// Descriptors a `declarations` entry may not set, because the loader generates them
/// (`validateLocalFontFunctionCall`).
const RESERVED_DECLARATIONS: [&str; 4] = ["src", "font-display", "font-weight", "font-style"];

/// One font call resolved to everything both the rewrite and the CSS need.
struct ResolvedCall {
    span: Span,
    /// `Inter` for a google font, the assigned const's name for a local one.
    family: String,
    /// The class-name slug: the google family for google fonts (so two modules
    /// importing `Inter` share one class, as they share one family), the const name for
    /// local ones (whose "binding" is the same `localFont` for every call).
    slug: String,
    variable: Option<String>,
    fallbacks: Vec<String>,
    adjust_fallback: bool,
    google: bool,
    local: Option<LocalFont>,
}

/// A module's `next/font` content: the import declarations to delete and the calls to
/// replace.
struct ResolvedModule {
    import_spans: Vec<Span>,
    calls: Vec<ResolvedCall>,
}

/// Parse a module and resolve every `next/font` call in it. `Ok(None)` when the module
/// imports no font factory at all — the cheap path every non-font module takes.
fn resolve_module(path: &Path, source: &str) -> Result<Option<ResolvedModule>, String> {
    let allocator = Allocator::default();
    let source_type = crate::parser::scan_source_type(path);
    let parsed = Parser::new(&allocator, source, source_type).parse();
    let program = &parsed.program;

    let (imports, import_spans) = collect_font_imports(program);
    if imports.is_empty() {
        return Ok(None);
    }
    let names: Vec<String> = imports.iter().map(|i| i.local.clone()).collect();
    let mut collector = CallCollector { names: &names, calls: Vec::new(), declared: Vec::new() };
    collector.visit_program(program);

    let mut resolved = Vec::new();
    for (name, call) in &collector.calls {
        let Some(import) = imports.iter().find(|i| &i.local == name) else { continue };
        let declared = collector
            .declared
            .iter()
            .find(|(span, _)| *span == call.span)
            .map(|(_, name)| name.clone());
        resolved.push(resolve_call(path, import, declared, &call.options, call.span)?);
    }
    Ok(Some(ResolvedModule { import_spans, calls: resolved }))
}

/// Apply Next's normalisation to one call's raw options.
fn resolve_call(
    path: &Path,
    import: &FontImport,
    declared: Option<String>,
    options: &RawOptions,
    span: Span,
) -> Result<ResolvedCall, String> {
    if import.google {
        return Ok(ResolvedCall {
            span,
            family: family_display(&import.local),
            slug: slug(&import.local),
            variable: options.variable.clone(),
            fallbacks: options.fallbacks.clone(),
            // Only `adjustFontFallback: false` turns the face off; Next treats every
            // other value as "generate it".
            adjust_fallback: options.adjust_bool.unwrap_or(true),
            google: true,
            local: None,
        });
    }

    // `next/font/local`: the family name IS the name of the const the call is assigned
    // to (Next's `variableName`), so a call that is not assigned to one has no family.
    let family = declared.ok_or_else(|| {
        format!(
            "{}: a `next/font/local` call must be assigned to a variable — Next derives \
             the font's family name from that name (it is what ends up in `font-family`), \
             so `localFont({{ ... }})` used inline has no family to emit. Write \
             `const myFont = localFont({{ ... }})`.",
            path.display()
        )
    })?;

    let display = options.display.clone().unwrap_or_else(|| "swap".to_string());
    if !ALLOWED_DISPLAY.contains(&display.as_str()) {
        return Err(format!(
            "{}: `next/font/local` got `display: \"{display}\"`. Available values: {}.",
            path.display(),
            ALLOWED_DISPLAY.join(", "),
        ));
    }
    for (prop, _) in &options.declarations {
        if RESERVED_DECLARATIONS.contains(&prop.as_str()) {
            return Err(format!(
                "{}: `next/font/local` declaration `{prop}` is generated by the loader \
                 itself and cannot be overridden through `declarations` (use the `src`, \
                 `display`, `weight` and `style` options).",
                path.display(),
            ));
        }
    }

    // Checked BEFORE "missing src", because a `src` written as anything other than the
    // two literal shapes leaves both of them empty and the reason would otherwise be
    // reported as absence.
    if options.src_unsupported {
        return Err(format!(
            "{}: `next/font/local`'s `src` must be a string literal or an array of \
             `{{ path: \"...\" }}` object literals — diffpack reads the font file at BUILD \
             time (to emit it and to derive the fallback metrics), so a computed `src` \
             cannot be followed.",
            path.display()
        ));
    }
    // `validateLocalFontFunctionCall`: a bare `src: "./x.woff2"` becomes ONE entry
    // carrying the call's top-level weight/style; an array's entries carry their own.
    let src = match (&options.src_string, &options.src_array) {
        (Some(single), _) => vec![LocalSrc {
            path: single.clone(),
            weight: options.weight.clone(),
            style: options.style.clone(),
        }],
        (None, Some(entries)) if !entries.is_empty() => entries.clone(),
        (None, Some(_)) => {
            return Err(format!("{}: `next/font/local` got an empty `src` array.", path.display()));
        }
        (None, None) => {
            return Err(format!(
                "{}: `next/font/local` is missing the required `src` option. It names the \
                 font FILE to load (`src: \"./My-Font.woff2\"`, or an array of \
                 `{{ path, weight, style }}` descriptors).",
                path.display()
            ));
        }
    };
    for entry in &src {
        font_format(&entry.path).ok_or_else(|| {
            format!(
                "{}: `next/font/local` got `src: \"{}\"`. Expected a .woff2, .woff, .ttf, \
                 .otf or .eot file.",
                path.display(),
                entry.path,
            )
        })?;
    }

    let adjust = match (options.adjust_bool, options.adjust_string.as_deref()) {
        (Some(false), _) => LocalAdjust::Off,
        (_, Some("Times New Roman")) => LocalAdjust::Serif,
        (_, Some("Arial")) | (_, None) | (Some(true), _) => LocalAdjust::SansSerif,
        (_, Some(other)) => {
            return Err(format!(
                "{}: `next/font/local` got `adjustFontFallback: \"{other}\"`. Available \
                 values: false, \"Arial\", \"Times New Roman\".",
                path.display(),
            ));
        }
    };

    // `weight: src.length === 1 ? src[0].weight : undefined` — the object (and the
    // generated class) only carry a weight when the family has exactly one file.
    let (weight, style) = if src.len() == 1 {
        (src[0].weight.clone(), src[0].style.clone())
    } else {
        (None, None)
    };

    Ok(ResolvedCall {
        span,
        slug: slug(&family),
        family,
        variable: options.variable.clone(),
        fallbacks: options.fallbacks.clone(),
        adjust_fallback: adjust != LocalAdjust::Off,
        google: false,
        local: Some(LocalFont {
            module: path.to_path_buf(),
            src,
            display,
            preload: options.preload.unwrap_or(true),
            declarations: options.declarations.clone(),
            adjust,
            weight,
            style,
        }),
    })
}

/// The `format()` string `next/font/local` writes for a file extension
/// (`validateLocalFontFunctionCall`'s `extToFormat`), or `None` for an extension it
/// does not accept.
fn font_format(path: &str) -> Option<&'static str> {
    let extension = path.rsplit('.').next()?.to_ascii_lowercase();
    match extension.as_str() {
        "woff" => Some("woff"),
        "woff2" => Some("woff2"),
        "ttf" => Some("truetype"),
        "otf" => Some("opentype"),
        "eot" => Some("embedded-opentype"),
        _ => None,
    }
}

/// The family list a `next/font` object and its CSS class both carry:
/// `'Inter', 'Inter Fallback', <user fallbacks…>` — exactly Next's
/// `postcss-next-font` `formattedFontFamilies`.
fn font_stack(family: &str, adjust_fallback: bool, fallbacks: &[String]) -> String {
    let mut families = vec![format!("'{family}'")];
    if adjust_fallback {
        families.push(format!("'{family} Fallback'"));
    }
    families.extend(fallbacks.iter().cloned());
    families.join(", ")
}

/// The `{ className, variable, style }` literal that replaces a font call, matching
/// what `next/font` produces at build time.
fn font_object(call: &ResolvedCall) -> String {
    let stack = font_stack(&call.family, call.adjust_fallback, &call.fallbacks);
    let mut style = vec![format!("fontFamily: {}", quote(&stack))];
    if let Some(local) = &call.local {
        // `postcss-next-font` puts the weight on the style object as a NUMBER
        // (`parseInt`), and only when the family resolved to a single file.
        if let Some(weight) = &local.weight {
            match weight.parse::<i64>() {
                Ok(number) => style.push(format!("fontWeight: {number}")),
                Err(_) => style.push(format!("fontWeight: {}", quote(weight))),
            }
        }
        if let Some(font_style) = &local.style {
            style.push(format!("fontStyle: {}", quote(font_style)));
        }
    }
    format!(
        "{{ className: {}, variable: {}, style: {{ {} }} }}",
        quote(&format!("__df_font_{}", call.slug)),
        quote(&format!("__df_fontvar_{}", call.slug)),
        style.join(", "),
    )
}

/// Rewrites a module's `next/font` calls into static objects and removes the
/// `next/font/*` imports, or `Ok(None)` when the module uses no `next/font`.
pub fn transform_next_font(path: &Path, source: &str) -> Result<Option<String>, String> {
    if !source.contains("next/font") {
        return Ok(None);
    }
    let Some(module) = resolve_module(path, source)? else {
        return Ok(None);
    };
    let mut edits: Vec<(Span, String)> = Vec::new();
    for span in module.import_spans {
        edits.push((span, String::new()));
    }
    for call in &module.calls {
        edits.push((call.span, font_object(call)));
    }
    Ok(Some(apply_edits(source, String::new(), edits)))
}

/// Scans a module for its `next/font` usages (family + `variable` option +
/// deterministic class names), for the app-router adapter to generate the matching
/// CSS. Mirrors [`transform_next_font`]'s naming so the classes agree.
pub fn scan_next_font(path: &Path, source: &str) -> Result<Vec<FontUsage>, String> {
    if !source.contains("next/font") {
        return Ok(Vec::new());
    }
    let Some(module) = resolve_module(path, source)? else {
        return Ok(Vec::new());
    };
    Ok(module
        .calls
        .into_iter()
        .map(|call| FontUsage {
            class_name: format!("__df_font_{}", call.slug),
            variable_class: format!("__df_fontvar_{}", call.slug),
            family: call.family,
            variable: call.variable,
            google: call.google,
            fallbacks: call.fallbacks,
            adjust_fallback: call.adjust_fallback,
            local: call.local,
        })
        .collect())
}

/// Generates the CSS for a set of font usages, plus the font files the build must emit
/// and the hrefs that want preloading.
///
/// For google usages: one Google Fonts `@import` covering all google families and the
/// metric-matched `"<family> Fallback"` `@font-face` per family, whose numbers come
/// from the app's OWN `next` install (`next/dist/server/capsize-font-metrics.json`) —
/// the same table `next build` reads, so they cannot drift from the oracle.
///
/// For local usages: one `@font-face` per `src` entry pointing at a content-hashed copy
/// of the real file under [`FONT_ASSET_DIR`], and a `"<family> Fallback"` face whose
/// overrides are computed from the font BINARY exactly as Next's fontkit pass does.
///
/// Then, for every usage, a `.className { font-family }` and a
/// `.variableClass { --var: family }` for each usage that declared a `variable`.
///
/// A family whose metrics cannot be derived is a hard error naming the family and the
/// file: shipping the page without the face is the silent layout shift `next/font`
/// exists to prevent.
pub fn generate(root: &Path, usages: &[FontUsage], asset_base: &str) -> Result<FontOutput, String> {
    let mut output = FontOutput::default();
    if usages.is_empty() {
        return Ok(output);
    }
    let css = &mut output.css;
    let mut families: Vec<String> = usages
        .iter()
        .filter(|u| u.google)
        .map(|u| u.family.replace(' ', "+"))
        .collect();
    families.sort();
    families.dedup();
    if !families.is_empty() {
        let query = families
            .iter()
            .map(|f| format!("family={f}:wght@100..900"))
            .collect::<Vec<_>>()
            .join("&");
        css.push_str(&format!(
            "@import url(\"https://fonts.googleapis.com/css2?{query}&display=swap\");\n"
        ));
    }

    // The metrics table is read at most once per generation, and only when some GOOGLE
    // usage actually asks for a metric-matched fallback.
    let mut metrics: Option<MetricsTable> = None;
    // Duplicates are collapsed by the RULE TEXT, not by family name. Real apps call the
    // same font in several modules (cal.com's `calFont` is constructed in three), and
    // those calls are the same usage only when they generate the same CSS. Keying on the
    // family instead would silently drop a second face that differs — a different
    // `display`, a different file — leaving the page rendering something the module that
    // lost never asked for.
    let mut emitted: Vec<String> = Vec::new();
    for usage in usages {
        let rules = match &usage.local {
            None => {
                if !usage.adjust_fallback {
                    continue;
                }
                let table = match &metrics {
                    Some(table) => table,
                    None => metrics.insert(MetricsTable::load(root)?),
                };
                vec![table.size_adjust_values(&usage.family)?.font_face(&usage.family)]
            }
            Some(local) => {
                let faces = local_font_faces(usage, local, asset_base, &mut output.assets)?;
                for href in faces.preloads {
                    if !output.preloads.contains(&href) {
                        output.preloads.push(href);
                    }
                }
                faces.rules
            }
        };
        for rule in rules {
            if emitted.contains(&rule) {
                continue;
            }
            css.push_str(&rule);
            emitted.push(rule);
        }
    }

    for usage in usages {
        let stack = font_stack(&usage.family, usage.adjust_fallback, &usage.fallbacks);
        let mut declarations = vec![format!("font-family: {stack};")];
        if let Some(local) = &usage.local {
            // Next's generated class carries the family's weight/style when the family
            // resolved to a single file, so `className` alone renders it correctly.
            if let Some(weight) = &local.weight {
                declarations.push(format!("font-weight: {weight};"));
            }
            if let Some(style) = &local.style {
                declarations.push(format!("font-style: {style};"));
            }
        }
        let mut block = format!(".{} {{ {} }}\n", usage.class_name, declarations.join(" "));
        if let Some(variable) = &usage.variable {
            block.push_str(&format!(".{} {{ {variable}: {stack}; }}\n", usage.variable_class));
        }
        if emitted.contains(&block) {
            continue;
        }
        css.push_str(&block);
        emitted.push(block);
    }
    Ok(output)
}

/// The `@font-face` rules for one `next/font/local` usage plus the preload hrefs, with
/// each source file recorded in `assets` for the build to copy.
struct LocalFaces {
    /// One `@font-face` rule per string, so identical rules coming from different
    /// modules collapse individually.
    rules: Vec<String>,
    preloads: Vec<String>,
}

fn local_font_faces(
    usage: &FontUsage,
    local: &LocalFont,
    asset_base: &str,
    assets: &mut Vec<FontAsset>,
) -> Result<LocalFaces, String> {
    let mut faces = LocalFaces { rules: Vec::new(), preloads: Vec::new() };
    // fontkit metadata per src entry, kept so the fallback face can be built from the
    // file Next would pick.
    let mut read: Vec<(&LocalSrc, crate::font_file::FontMetrics)> = Vec::new();
    let custom_family = local
        .declarations
        .iter()
        .find(|(prop, _)| prop == "font-family")
        .map(|(_, value)| value.clone());

    for entry in &local.src {
        let source = resolve_src(local, &entry.path)?;
        let bytes = std::fs::read(&source).map_err(|error| {
            format!("cannot read font file {}: {error}", source.display())
        })?;
        let metrics = crate::font_file::read_metrics_from_bytes(&bytes, &source).map_err(
            |error| {
                format!(
                    "{}: `next/font/local` cannot read the metrics of `{}`: {error}. Next \
                     derives the size-adjusted fallback face from the font binary, and \
                     without it the page reflows when the font loads.",
                    local.module.display(),
                    entry.path,
                )
            },
        )?;
        let file = asset_file_name(&source, &bytes)?;
        let url = format!("{asset_base}/{FONT_ASSET_DIR}/{file}");
        let asset = FontAsset { source: source.clone(), file: format!("{FONT_ASSET_DIR}/{file}") };
        if !assets.contains(&asset) {
            assets.push(asset);
        }
        if local.preload {
            faces.preloads.push(url.clone());
        }

        // The loader's property ORDER: `declarations` first, then the generated
        // font-family (unless the declarations already set one), src, font-display,
        // then weight and style when they are known.
        let format = font_format(&entry.path).expect("validated above");
        let mut properties: Vec<String> = local
            .declarations
            .iter()
            .map(|(prop, value)| format!("{prop}: {value};"))
            .collect();
        if custom_family.is_none() {
            properties.push(format!("font-family: {};", usage.family));
        }
        properties.push(format!("src: url({url}) format('{format}');"));
        properties.push(format!("font-display: {};", local.display));
        if let Some(weight) = &entry.weight {
            properties.push(format!("font-weight: {weight};"));
        }
        if let Some(style) = &entry.style {
            properties.push(format!("font-style: {style};"));
        }
        faces.rules.push(format!("@font-face {{ {} }}\n", properties.join(" ")));
        read.push((entry, metrics));
    }

    if local.adjust != LocalAdjust::Off {
        let chosen = pick_fallback_source(&read);
        let (_, metrics) = read[chosen];
        let overrides = SizeAdjustValues::from_font_metrics(&metrics, local.adjust);
        faces.rules.push(overrides.font_face(&usage.family));
    }
    Ok(faces)
}

/// Resolve a `src` path against the module the call is written in, the way Next's
/// loader `resolve` does. A path that does not exist is a hard error naming both.
fn resolve_src(local: &LocalFont, src: &str) -> Result<PathBuf, String> {
    let directory = local.module.parent().unwrap_or(Path::new("."));
    let joined = directory.join(src);
    // Lexical normalisation, so the error message shows the path the app meant rather
    // than one with `..` still in it.
    let mut normalized = PathBuf::new();
    for component in joined.components() {
        match component {
            std::path::Component::ParentDir => {
                normalized.pop();
            }
            std::path::Component::CurDir => {}
            other => normalized.push(other.as_os_str()),
        }
    }
    if normalized.is_file() {
        return Ok(normalized);
    }
    Err(format!(
        "{}: `next/font/local` cannot find `{src}` (resolved to {}). `src` is resolved \
         relative to the module the call is written in, and diffpack must read the file \
         at build time to emit it and to derive the fallback metrics.",
        local.module.display(),
        normalized.display(),
    ))
}

/// The served file name for an emitted font: the source's stem, a content hash, and the
/// original extension. Content-addressed so a font that changes gets a new URL and one
/// that does not keeps its cache entry.
fn asset_file_name(source: &Path, bytes: &[u8]) -> Result<String, String> {
    let stem = source
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| format!("font file {} has no usable name", source.display()))?;
    let extension = source
        .extension()
        .and_then(|s| s.to_str())
        .ok_or_else(|| format!("font file {} has no extension", source.display()))?;
    let stem: String = stem
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    Ok(format!("{stem}-{:016x}.{}", crate::bundler::content_hash(bytes), extension.to_lowercase()))
}

/// Next's `pickFontFileForFallbackGeneration`: of a family's files, the one most likely
/// to carry the bulk of a page's text — closest to weight 400, normal style preferred,
/// the thinner one on a tie. Returns the index into `read`.
fn pick_fallback_source(read: &[(&LocalSrc, crate::font_file::FontMetrics)]) -> usize {
    let mut used = 0usize;
    for candidate in 1..read.len() {
        let used_distance = distance_from_normal_weight(read[used].0.weight.as_deref());
        let candidate_distance = distance_from_normal_weight(read[candidate].0.weight.as_deref());
        let style = read[candidate].0.style.as_deref();
        if used_distance == candidate_distance && matches!(style, None | Some("normal")) {
            used = candidate;
            continue;
        }
        if candidate_distance.abs() < used_distance.abs() {
            used = candidate;
            continue;
        }
        if candidate_distance.abs() == used_distance.abs() && candidate_distance < used_distance {
            used = candidate;
        }
    }
    used
}

/// Next's `getDistanceFromNormalWeight`: how far a `font-weight` value sits from 400,
/// with a variable font's `"100 900"` range counting as 0 when it contains 400.
fn distance_from_normal_weight(weight: Option<&str>) -> f64 {
    const NORMAL: f64 = 400.0;
    let Some(weight) = weight else { return 0.0 };
    let number = |value: &str| -> f64 {
        match value {
            "normal" => NORMAL,
            "bold" => 700.0,
            other => other.parse::<f64>().unwrap_or(f64::NAN),
        }
    };
    let mut parts = weight.split_whitespace().map(number);
    let first = parts.next().unwrap_or(f64::NAN);
    let Some(second) = parts.next() else {
        return first - NORMAL;
    };
    if first <= NORMAL && second >= NORMAL {
        return 0.0;
    }
    let first_distance = first - NORMAL;
    let second_distance = second - NORMAL;
    if first_distance.abs() < second_distance.abs() { first_distance } else { second_distance }
}
/// The four `@font-face` descriptors that make a locally-installed face occupy the
/// same space as the webfont, as percentages already formatted the way Next formats
/// them (`"90.44"`), plus the local family the face is built from (`Arial`).
#[derive(Debug, Clone, PartialEq, Eq)]
struct SizeAdjustValues {
    fallback_font: String,
    ascent: String,
    descent: String,
    line_gap: String,
    size_adjust: String,
}

impl SizeAdjustValues {
    /// The `@font-face` Next generates for `<family> Fallback`
    /// (`postcss-next-font`'s `fallbackFontFace`).
    fn font_face(&self, family: &str) -> String {
        format!(
            "@font-face {{ font-family: '{family} Fallback'; src: local(\"{}\"); \
             ascent-override: {}%; descent-override: {}%; line-gap-override: {}%; \
             size-adjust: {}%; }}\n",
            self.fallback_font, self.ascent, self.descent, self.line_gap, self.size_adjust
        )
    }

    /// Next's `getFallbackMetricsFromFontFile`
    /// (`@next/font/dist/local/get-fallback-metrics-from-font-file.js`), number for
    /// number: the fallback face is Arial (or Times New Roman when the call asked for
    /// it), scaled so its average character width matches the real font's, with the
    /// three vertical overrides divided by that same scale.
    ///
    /// The two fallback constants below are Next's own, measured with fontkit on the
    /// system files and hard-coded in that module — NOT the capsize table the google
    /// loader reads, which is keyed by google family name and has no entry for an app's
    /// private font. A font whose average width could not be measured (some sample
    /// character has no glyph) gets `sizeAdjust = 1`, which is Next's documented
    /// behaviour rather than a diffpack substitution.
    fn from_font_metrics(
        metrics: &crate::font_file::FontMetrics,
        adjust: LocalAdjust,
    ) -> Self {
        // (name, azAvgWidth, unitsPerEm) exactly as Next hard-codes them.
        let (fallback_font, fallback_avg_width, fallback_units) = match adjust {
            LocalAdjust::Serif => ("Times New Roman", 854.3953488372093, 2048.0),
            _ => ("Arial", 934.5116279069767, 2048.0),
        };
        let fallback_avg = fallback_avg_width / fallback_units;
        let size_adjust = match metrics.az_avg_width {
            Some(width) if width != 0.0 => width / metrics.units_per_em / fallback_avg,
            _ => 1.0,
        };
        let percent = |value: f64| format!("{:.2}", (value * 100.0).abs());
        Self {
            fallback_font: fallback_font.to_string(),
            ascent: percent(metrics.ascent / (metrics.units_per_em * size_adjust)),
            descent: percent(metrics.descent / (metrics.units_per_em * size_adjust)),
            line_gap: percent(metrics.line_gap / (metrics.units_per_em * size_adjust)),
            size_adjust: percent(size_adjust),
        }
    }
}

/// Next's precalculated font metrics, read from the app's own `next` install.
struct MetricsTable {
    path: PathBuf,
    entries: serde_json::Value,
}

impl MetricsTable {
    /// Locates `next/dist/server/capsize-font-metrics.json` from the project root,
    /// walking up so a workspace/hoisted `node_modules` is found too.
    fn load(root: &Path) -> Result<Self, String> {
        const RELATIVE: &str = "node_modules/next/dist/server/capsize-font-metrics.json";
        let mut searched = Vec::new();
        for dir in root.ancestors() {
            let candidate = dir.join(RELATIVE);
            if candidate.is_file() {
                let text = std::fs::read_to_string(&candidate)
                    .map_err(|error| format!("cannot read {}: {error}", candidate.display()))?;
                let entries: serde_json::Value = serde_json::from_str(&text)
                    .map_err(|error| format!("cannot parse {}: {error}", candidate.display()))?;
                return Ok(Self { path: candidate, entries });
            }
            searched.push(candidate.display().to_string());
        }
        Err(format!(
            "next/font: cannot find Next's font-metrics table ({RELATIVE}) from {}. \
             `next/font` derives the size-adjusted fallback face — the whole reason it \
             exists, since without it the page reflows when the webfont loads — from that \
             table. Install the app's dependencies so `next` is present. Searched: {}",
            root.display(),
            searched.join(", "),
        ))
    }

    /// Next's `calculateSizeAdjustValues` (`next/dist/server/font-utils.js`), number
    /// for number: the fallback is Arial for every non-serif family and Times New
    /// Roman for serif ones, scaled so its average character width matches the real
    /// family's.
    fn size_adjust_values(&self, family: &str) -> Result<SizeAdjustValues, String> {
        let key = format_name(family);
        let entry = self.entry(&key).ok_or_else(|| {
            format!(
                "next/font: no font metrics for `{family}` (key `{key}`) in {}. Next \
                 generates the size-adjusted `{family} Fallback` face from this table, and \
                 shipping the family without it is the layout shift next/font exists to \
                 prevent. Check the family name spelled in the `next/font/google` import, \
                 or pass `adjustFontFallback: false` to opt out deliberately.",
                self.path.display()
            )
        })?;
        let number = |name: &str| -> Result<f64, String> {
            entry.get(name).and_then(serde_json::Value::as_f64).ok_or_else(|| {
                format!(
                    "next/font: metrics entry `{key}` in {} has no numeric `{name}`",
                    self.path.display()
                )
            })
        };
        let category = entry.get("category").and_then(serde_json::Value::as_str).unwrap_or("");
        let ascent = number("ascent")?;
        let descent = number("descent")?;
        let line_gap = number("lineGap")?;
        let units_per_em = number("unitsPerEm")?;
        let x_width_avg = number("xWidthAvg")?;

        let fallback_font = if category == "serif" { "Times New Roman" } else { "Arial" };
        let fallback_key = format_name(fallback_font);
        let fallback = self.entry(&fallback_key).ok_or_else(|| {
            format!(
                "next/font: the metrics table {} carries no `{fallback_key}` entry, so the \
                 fallback face for `{family}` cannot be scaled",
                self.path.display()
            )
        })?;
        let fallback_number = |name: &str| -> Result<f64, String> {
            fallback.get(name).and_then(serde_json::Value::as_f64).ok_or_else(|| {
                format!(
                    "next/font: metrics entry `{fallback_key}` in {} has no numeric `{name}`",
                    self.path.display()
                )
            })
        };
        let fallback_avg = fallback_number("xWidthAvg")? / fallback_number("unitsPerEm")?;
        let size_adjust =
            if x_width_avg != 0.0 { (x_width_avg / units_per_em) / fallback_avg } else { 1.0 };

        // Next's `formatOverrideValue`: `Math.abs(val * 100).toFixed(2)` — descent is
        // negative in the table and is emitted as a positive percentage.
        let percent = |value: f64| format!("{:.2}", (value * 100.0).abs());
        Ok(SizeAdjustValues {
            fallback_font: fallback_font.to_string(),
            ascent: percent(ascent / (units_per_em * size_adjust)),
            descent: percent(descent / (units_per_em * size_adjust)),
            line_gap: percent(line_gap / (units_per_em * size_adjust)),
            size_adjust: percent(size_adjust),
        })
    }

    fn entry(&self, key: &str) -> Option<&serde_json::Value> {
        self.entries.get(key).filter(|value| value.is_object())
    }
}

/// Next's `formatName` (`next/dist/server/font-utils.js`): the table is keyed by a
/// camel-cased, space-stripped family name (`"Times New Roman"` -> `timesNewRoman`,
/// `"Inter"` -> `inter`).
fn format_name(family: &str) -> String {
    let chars: Vec<char> = family.chars().collect();
    let word = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let mut out = String::with_capacity(family.len());
    for (index, &c) in chars.iter().enumerate() {
        // JS: /(?:^\w|[A-Z]|\b\w)/g — the first character, any uppercase letter, or a
        // word character starting a word. The first match lowercases, the rest uppercase.
        let boundary = word(c) && (index == 0 || !word(chars[index - 1]));
        let matched = boundary || c.is_ascii_uppercase();
        if matched && index == 0 {
            out.extend(c.to_lowercase());
        } else if matched {
            out.extend(c.to_uppercase());
        } else {
            out.push(c);
        }
    }
    out.chars().filter(|c| !c.is_whitespace()).collect()
}


/// Record the local font files the generated CSS points at, under the Next adapter
/// directory, so the client emit can copy them without re-walking the project.
///
/// Written on EVERY configure pass, including when the app has no local fonts (an empty
/// list), so a stale manifest from a previous build can never make the emit copy a font
/// the current CSS does not reference.
pub fn write_font_manifest(adapter_dir: &Path, assets: &[FontAsset]) -> Result<(), String> {
    let entries: Vec<serde_json::Value> = assets
        .iter()
        .map(|asset| {
            serde_json::json!({ "source": asset.source.to_string_lossy(), "file": asset.file })
        })
        .collect();
    let path = adapter_dir.join(FONT_MANIFEST_FILE);
    let text = serde_json::to_string_pretty(&serde_json::Value::Array(entries))
        .map_err(|error| format!("cannot serialize {}: {error}", path.display()))?;
    std::fs::write(&path, text)
        .map_err(|error| format!("cannot write {}: {error}", path.display()))
}

/// Copy every local font file the last configure pass recorded into the served
/// `public/` output, at the content-hashed name the emitted CSS asks for. Returns how
/// many files were written; a no-op for a project with no `next/font/local`.
///
/// A recorded font that has since vanished is a hard error: the CSS in the very same
/// build references its URL, so serving the page without it would render the app in a
/// font it never asked for — the exact failure the old "not implemented" refusal
/// existed to prevent.
pub fn emit_font_assets(root: &Path, out_public: &Path) -> Result<usize, String> {
    let manifest = root.join(crate::next_adapter::ADAPTER_DIR).join(FONT_MANIFEST_FILE);
    let Ok(text) = std::fs::read_to_string(&manifest) else {
        return Ok(0);
    };
    let entries: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| format!("cannot parse {}: {error}", manifest.display()))?;
    let Some(entries) = entries.as_array() else {
        return Err(format!("{}: expected a JSON array", manifest.display()));
    };
    let mut written = 0usize;
    for entry in entries {
        let (Some(source), Some(file)) = (
            entry.get("source").and_then(serde_json::Value::as_str),
            entry.get("file").and_then(serde_json::Value::as_str),
        ) else {
            return Err(format!("{}: entry is missing `source`/`file`", manifest.display()));
        };
        let destination = out_public.join(file);
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
        }
        std::fs::copy(source, &destination).map_err(|error| {
            format!(
                "cannot copy the `next/font/local` file {source} to {}: {error}. The \
                 generated @font-face already points at {file}, so the page would load a \
                 font that is not there.",
                destination.display(),
            )
        })?;
        written += 1;
    }
    Ok(written)
}
#[cfg(test)]
mod tests {
    use super::*;

    fn t(source: &str) -> Option<String> {
        transform_next_font(Path::new("app/layout.tsx"), source).unwrap()
    }

    /// A throwaway project root carrying a `next` install whose metrics table holds the
    /// REAL `Inter` and `Arial` entries copied out of
    /// `next/dist/server/capsize-font-metrics.json`, so the generated face can be
    /// compared with what `next build` actually emits for the same app.
    fn scaffold_metrics(name: &str) -> std::path::PathBuf {
        let mut root = std::env::temp_dir();
        root.push(format!("diffpack-next-font-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = root.join("node_modules/next/dist/server");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("capsize-font-metrics.json"),
            r#"{
              "inter": { "familyName": "Inter", "category": "sans-serif", "ascent": 1984,
                         "descent": -494, "lineGap": 0, "unitsPerEm": 2048, "xWidthAvg": 978 },
              "arial": { "familyName": "Arial", "category": "sans-serif", "ascent": 1854,
                         "descent": -434, "lineGap": 67, "unitsPerEm": 2048, "xWidthAvg": 913 }
            }"#,
        )
        .unwrap();
        root
    }

    fn scan(source: &str) -> Vec<FontUsage> {
        scan_next_font(Path::new("app/layout.tsx"), source).unwrap()
    }

    /// The CSS half of a generation, for the google cases where no asset is emitted.
    fn generate_css(root: &Path, usages: &[FontUsage]) -> Result<String, String> {
        generate(root, usages, "").map(|output| output.css)
    }

    #[test]
    fn rewrites_google_font_calls_and_drops_the_import() {
        let out = t("import { Geist, Geist_Mono } from \"next/font/google\";\nconst a = Geist({ variable: \"--font-geist-sans\", subsets: [\"latin\"] });\nconst b = Geist_Mono({ variable: \"--font-geist-mono\", subsets: [\"latin\"] });\n").unwrap();
        assert!(!out.contains("next/font/google"), "import must be removed: {out}");
        assert!(!out.contains("Geist("), "the throwing call must be gone: {out}");
        assert!(out.contains("className: \"__df_font_geist\""), "{out}");
        assert!(out.contains("variable: \"__df_fontvar_geist\""), "{out}");
        assert!(out.contains("__df_font_geist_mono"), "{out}");
        assert!(out.contains("fontFamily: \"'Geist', "), "{out}");
    }

    #[test]
    fn plain_modules_are_untouched() {
        assert_eq!(t("export const x = 1;"), None);
        assert_eq!(t("import Image from \"next/image\";\nexport default Image;"), None);
    }

    #[test]
    fn generates_google_import_and_variable_classes() {
        let root = scaffold_metrics("variable-classes");
        std::fs::write(
            root.join("node_modules/next/dist/server/capsize-font-metrics.json"),
            r#"{
              "geist": { "category": "sans-serif", "ascent": 1005, "descent": -295,
                         "lineGap": 0, "unitsPerEm": 1000, "xWidthAvg": 467 },
              "geistMono": { "category": "monospace", "ascent": 1005, "descent": -295,
                             "lineGap": 0, "unitsPerEm": 1000, "xWidthAvg": 600 },
              "arial": { "category": "sans-serif", "ascent": 1854, "descent": -434,
                         "lineGap": 67, "unitsPerEm": 2048, "xWidthAvg": 913 }
            }"#,
        )
        .unwrap();
        let usages = scan(
            "import { Geist, Geist_Mono } from \"next/font/google\";\nconst a = Geist({ variable: \"--font-geist-sans\" });\nconst b = Geist_Mono({ variable: \"--font-geist-mono\" });\n",
        );
        assert_eq!(usages.len(), 2);
        let css = generate_css(&root, &usages).unwrap();
        assert!(css.contains("fonts.googleapis.com/css2?family=Geist:wght"), "{css}");
        assert!(css.contains("family=Geist+Mono:wght"), "{css}");
        assert!(css.contains(".__df_fontvar_geist { --font-geist-sans: 'Geist'"), "{css}");
        assert!(css.contains(".__df_font_geist { font-family: 'Geist'"), "{css}");
    }

    /// FINDINGS #18. `next/font`'s whole purpose is that the fallback face occupies the
    /// SAME space as the webfont, so nothing reflows while it loads. Next emits, for
    /// `Inter({ subsets: ["latin"] })`:
    ///
    /// ```css
    /// @font-face{font-family:Inter Fallback;src:local(Arial);ascent-override:90.44%;
    ///            descent-override:22.52%;line-gap-override:0.0%;size-adjust:107.12%}
    /// .className{font-family:Inter,Inter Fallback}
    /// ```
    ///
    /// (verified against `next build`'s own CSS for integration/e2e/apps/next-blog-starter).
    /// A generic `ui-sans-serif, system-ui, …` stack instead is the layout shift the
    /// feature exists to prevent — pages measured 5px taller.
    #[test]
    fn emits_the_metric_matched_fallback_face_next_generates() {
        let root = scaffold_metrics("fallback-face");
        let usages = scan(
            "import { Inter } from \"next/font/google\";\nconst inter = Inter({ subsets: [\"latin\"] });\n",
        );
        assert_eq!(usages.len(), 1);
        assert!(usages[0].adjust_fallback);
        let css = generate_css(&root, &usages).unwrap();
        assert!(
            css.contains(
                "@font-face { font-family: 'Inter Fallback'; src: local(\"Arial\"); \
                 ascent-override: 90.44%; descent-override: 22.52%; line-gap-override: 0.00%; \
                 size-adjust: 107.12%; }"
            ),
            "the size-adjusted fallback face must match `next build`'s: {css}"
        );
        assert!(
            css.contains(".__df_font_inter { font-family: 'Inter', 'Inter Fallback'; }"),
            "{css}"
        );
        assert!(
            !css.contains("ui-sans-serif"),
            "Next appends no generic stack — only the metric-matched face: {css}"
        );
        // The object the macro leaves behind carries the same family list, so
        // `inter.style.fontFamily` agrees with the class.
        let out = t("import { Inter } from \"next/font/google\";\nconst inter = Inter({ subsets: [\"latin\"] });\n").unwrap();
        assert!(out.contains("fontFamily: \"'Inter', 'Inter Fallback'\""), "{out}");
    }

    /// `adjustFontFallback: false` is Next's documented opt-out, and `fallback: [...]`
    /// appends the app's own families after the generated one.
    #[test]
    fn honors_adjust_font_fallback_false_and_the_fallback_list() {
        let root = scaffold_metrics("opt-out");
        let usages = scan(
            "import { Inter } from \"next/font/google\";\nconst inter = Inter({ adjustFontFallback: false, fallback: [\"Helvetica\", \"sans-serif\"] });\n",
        );
        assert!(!usages[0].adjust_fallback);
        assert_eq!(usages[0].fallbacks, vec!["Helvetica".to_string(), "sans-serif".to_string()]);
        let css = generate_css(&root, &usages).unwrap();
        assert!(!css.contains("@font-face"), "opted out, so no fallback face: {css}");
        assert!(
            css.contains(".__df_font_inter { font-family: 'Inter', Helvetica, sans-serif; }"),
            "{css}"
        );
    }

    /// A family the metrics table does not carry cannot get a metric-matched face, and
    /// shipping the page without one is the silent layout shift. It must be loud.
    #[test]
    fn an_unknown_family_is_a_hard_error_naming_it() {
        let root = scaffold_metrics("unknown-family");
        let usages = scan(
            "import { Not_A_Real_Font } from \"next/font/google\";\nconst f = Not_A_Real_Font({});\n",
        );
        let error = generate_css(&root, &usages).unwrap_err();
        assert!(error.contains("Not A Real Font"), "{error}");
        assert!(error.contains("notAReal"), "the table key must be named too: {error}");
        assert!(error.contains("adjustFontFallback: false"), "{error}");
    }

    /// A missing `next` install is named, not silently skipped.
    #[test]
    fn a_missing_metrics_table_is_a_hard_error() {
        let mut root = std::env::temp_dir();
        root.push(format!("diffpack-next-font-{}-no-next", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let usages = scan("import { Inter } from \"next/font/google\";\nconst i = Inter({});\n");
        let error = generate_css(&root, &usages).unwrap_err();
        assert!(error.contains("capsize-font-metrics.json"), "{error}");
    }

    // ---------------------------------------------------------------------------
    // next/font/local
    //
    // The oracle for every number below is `corepack yarn next build` on cal.com
    // (`apps/web/app/layout.tsx`), whose emitted stylesheet contains
    //
    // ```css
    // @font-face{font-family:calFont;src:url(../media/CalSans_SemiBold-s.p.<hash>.woff2)
    //            format("woff2");font-display:block;font-weight:600}
    // @font-face{font-family:calFont Fallback;src:local(Arial);ascent-override:98.6%;
    //            descent-override:19.72%;line-gap-override:0.0%;size-adjust:101.42%}
    // .<hash>__className{font-family:calFont,calFont Fallback;font-weight:600}
    // .<hash>__variable{--font-cal:"calFont", "calFont Fallback"}
    // ```
    //
    // — i.e. the family is the name of the CONST, the file becomes a build asset, and
    // the fallback face is scaled from the font binary's own metrics. Running
    // `getFallbackMetricsFromFontFile` over `CalSans-SemiBold.woff2` under Next's own
    // fontkit gives ascent 1000, descent -200, lineGap 0, unitsPerEm 1000 and an
    // average width of 462.7906976744186, which is exactly what `crate::font_file`
    // reads from the same file.
    // ---------------------------------------------------------------------------

    /// A project laid out like cal.com's: a module that calls `localFont` and the font
    /// file it points at. Returns (root, module path).
    fn scaffold_local(name: &str, source: &str, units_per_em: u16, advance: u16) -> PathBuf {
        let mut root = std::env::temp_dir();
        root.push(format!("diffpack-next-font-local-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("app")).unwrap();
        std::fs::create_dir_all(root.join("fonts")).unwrap();
        std::fs::write(root.join("app/layout.tsx"), source).unwrap();
        std::fs::write(
            root.join("fonts/CalSans-SemiBold.ttf"),
            crate::font_file::synthetic_font(units_per_em, advance),
        )
        .unwrap();
        root
    }

    fn scan_at(module: &Path) -> Vec<FontUsage> {
        let source = std::fs::read_to_string(module).unwrap();
        scan_next_font(module, &source).unwrap()
    }

    /// cal.com's exact call. The family name Next uses is the name of the const, and
    /// the rewritten object carries it plus the `weight` the call passed.
    #[test]
    fn rewrites_a_local_font_call_using_the_const_name_as_the_family() {
        let source = "import localFont from \"next/font/local\";\n\
             const calFont = localFont({ src: \"../fonts/CalSans-SemiBold.woff2\", \
             variable: \"--font-cal\", preload: true, display: \"block\", weight: \"600\" });\n";
        let out = transform_next_font(Path::new("app/layout.tsx"), source).unwrap().unwrap();
        assert!(!out.contains("next/font/local"), "the import must be removed: {out}");
        assert!(!out.contains("localFont("), "the throwing call must be gone: {out}");
        assert!(out.contains("className: \"__df_font_calfont\""), "{out}");
        assert!(out.contains("variable: \"__df_fontvar_calfont\""), "{out}");
        assert!(
            out.contains("fontFamily: \"'calFont', 'calFont Fallback'\""),
            "the family is the const's name, with the metric-matched face after it: {out}"
        );
        assert!(out.contains("fontWeight: 600"), "{out}");
    }

    /// Two local fonts in one module must not collide: the import binding is the same
    /// `localFont` for both, so the class name has to come from the const.
    #[test]
    fn two_local_fonts_get_distinct_classes() {
        let out = transform_next_font(
            Path::new("app/layout.tsx"),
            "import localFont from \"next/font/local\";\n\
             const a = localFont({ src: \"./a.woff2\" });\n\
             const b = localFont({ src: \"./b.woff2\" });\n",
        )
        .unwrap()
        .unwrap();
        assert!(out.contains("__df_font_a"), "{out}");
        assert!(out.contains("__df_font_b"), "{out}");
    }

    /// The whole emit for cal.com's shape: the real `@font-face` pointing at an emitted
    /// asset, the size-adjusted fallback face computed from the font binary, the class
    /// carrying family + weight, the CSS variable, the copied file and the preload.
    #[test]
    fn emits_the_face_asset_and_fallback_next_generates_for_a_local_font() {
        let root = scaffold_local(
            "calcom",
            "import localFont from \"next/font/local\";\n\
             const calFont = localFont({ src: \"../fonts/CalSans-SemiBold.ttf\", \
             variable: \"--font-cal\", preload: true, display: \"block\", weight: \"600\" });\n",
            1000,
            500,
        );
        let usages = scan_at(&root.join("app/layout.tsx"));
        assert_eq!(usages.len(), 1);
        assert_eq!(usages[0].family, "calFont");
        assert!(!usages[0].google);

        let output = generate(&root, &usages, "").unwrap();
        let css = &output.css;
        // One real face, pointing at the emitted asset, in Next's property order.
        assert!(
            css.contains("@font-face { font-family: calFont; src: url(/_diffpack-font/CalSans-SemiBold-"),
            "{css}"
        );
        assert!(css.contains(") format('truetype'); font-display: block; font-weight: 600; }"), "{css}");
        // The metric-matched face, scaled from the font's OWN metrics (upem 1000,
        // ascent 1000, descent -200, average width 500) against Next's hard-coded
        // Arial (azAvgWidth 934.5116279069767 / unitsPerEm 2048).
        assert!(
            css.contains(
                "@font-face { font-family: 'calFont Fallback'; src: local(\"Arial\"); \
                 ascent-override: 91.26%; descent-override: 18.25%; line-gap-override: 0.00%; \
                 size-adjust: 109.58%; }"
            ),
            "{css}"
        );
        assert!(
            css.contains(".__df_font_calfont { font-family: 'calFont', 'calFont Fallback'; font-weight: 600; }"),
            "{css}"
        );
        assert!(
            css.contains(".__df_fontvar_calfont { --font-cal: 'calFont', 'calFont Fallback'; }"),
            "{css}"
        );

        // The file itself is a build asset, and its URL is the one the face names.
        assert_eq!(output.assets.len(), 1);
        assert_eq!(output.assets[0].source, root.join("fonts/CalSans-SemiBold.ttf"));
        assert!(output.assets[0].file.starts_with("_diffpack-font/CalSans-SemiBold-"), "{:?}", output.assets);
        assert!(css.contains(&output.assets[0].file), "the face must name the emitted file: {css}");
        assert_eq!(output.preloads, vec![format!("/{}", output.assets[0].file)]);

        // And the emit copies exactly what the manifest recorded.
        let adapter = root.join(crate::next_adapter::ADAPTER_DIR);
        std::fs::create_dir_all(&adapter).unwrap();
        write_font_manifest(&adapter, &output.assets).unwrap();
        let out_public = root.join("out/public");
        assert_eq!(emit_font_assets(&root, &out_public).unwrap(), 1);
        assert!(out_public.join(&output.assets[0].file).is_file());
    }

    /// The asset URL is prefixed by `next.config`'s `assetPrefix`/`basePath`, like every
    /// other emitted asset — otherwise the face 404s on a deployment served under a path.
    #[test]
    fn the_font_url_carries_the_asset_prefix() {
        let root = scaffold_local(
            "asset-prefix",
            "import localFont from \"next/font/local\";\n\
             const calFont = localFont({ src: \"../fonts/CalSans-SemiBold.ttf\" });\n",
            1000,
            500,
        );
        let usages = scan_at(&root.join("app/layout.tsx"));
        let output = generate(&root, &usages, "/docs").unwrap();
        assert!(output.css.contains("src: url(/docs/_diffpack-font/CalSans-SemiBold-"), "{}", output.css);
        assert!(output.preloads[0].starts_with("/docs/_diffpack-font/"), "{:?}", output.preloads);
    }

    /// `src` as an ARRAY of descriptors: one `@font-face` per file, each with its own
    /// weight/style, and NO weight on the class (Next only puts one there when the
    /// family resolved to a single file). The fallback face is generated from the file
    /// `pickFontFileForFallbackGeneration` chooses — the one nearest weight 400.
    #[test]
    fn an_src_array_emits_one_face_per_file_and_picks_the_fallback_source() {
        let root = scaffold_local(
            "src-array",
            "import localFont from \"next/font/local\";\n\
             const myFont = localFont({ src: [\
               { path: \"../fonts/Bold.ttf\", weight: \"700\", style: \"normal\" },\
               { path: \"../fonts/Regular.ttf\", weight: \"400\", style: \"normal\" }\
             ], variable: \"--my\" });\n",
            1000,
            500,
        );
        // The two files differ, so the chosen one is observable in the fallback numbers.
        std::fs::write(root.join("fonts/Bold.ttf"), crate::font_file::synthetic_font(1000, 900))
            .unwrap();
        std::fs::write(root.join("fonts/Regular.ttf"), crate::font_file::synthetic_font(1000, 500))
            .unwrap();
        let usages = scan_at(&root.join("app/layout.tsx"));
        let output = generate(&root, &usages, "").unwrap();
        let css = &output.css;
        assert_eq!(css.matches("src: url(").count(), 2, "one face per src entry: {css}");
        assert!(css.contains("font-weight: 700; font-style: normal; }"), "{css}");
        assert!(css.contains("font-weight: 400; font-style: normal; }"), "{css}");
        assert_eq!(output.assets.len(), 2);
        // Regular (weight 400, average width 500) is the fallback source, so the
        // overrides are the same as the single-file case — NOT Bold's.
        assert!(css.contains("size-adjust: 109.58%;"), "{css}");
        // No weight on the class: the family has more than one file.
        assert!(
            css.contains(".__df_font_myfont { font-family: 'myFont', 'myFont Fallback'; }"),
            "{css}"
        );
    }

    /// `adjustFontFallback: 'Times New Roman'` scales the serif fallback instead, and
    /// `false` is the documented opt-out (no generated face at all).
    #[test]
    fn honors_the_local_adjust_font_fallback_option() {
        let root = scaffold_local(
            "serif",
            "import localFont from \"next/font/local\";\n\
             const serifFont = localFont({ src: \"../fonts/CalSans-SemiBold.ttf\", \
             adjustFontFallback: \"Times New Roman\" });\n",
            1000,
            500,
        );
        let css = generate(&root, &scan_at(&root.join("app/layout.tsx")), "").unwrap().css;
        assert!(
            css.contains(
                "font-family: 'serifFont Fallback'; src: local(\"Times New Roman\"); \
                 ascent-override: 83.44%; descent-override: 16.69%; line-gap-override: 0.00%; \
                 size-adjust: 119.85%;"
            ),
            "{css}"
        );

        let root = scaffold_local(
            "opt-out",
            "import localFont from \"next/font/local\";\n\
             const plainFont = localFont({ src: \"../fonts/CalSans-SemiBold.ttf\", \
             adjustFontFallback: false });\n",
            1000,
            500,
        );
        let usages = scan_at(&root.join("app/layout.tsx"));
        assert!(!usages[0].adjust_fallback);
        let css = generate(&root, &usages, "").unwrap().css;
        assert!(!css.contains("Fallback"), "opted out, so no generated face: {css}");
        assert_eq!(css.matches("@font-face").count(), 1, "{css}");
    }

    /// `preload: false` suppresses the head `<link rel="preload">` (the file is still
    /// emitted — the CSS references it either way).
    #[test]
    fn preload_false_emits_the_asset_but_no_preload_link() {
        let root = scaffold_local(
            "no-preload",
            "import localFont from \"next/font/local\";\n\
             const quietFont = localFont({ src: \"../fonts/CalSans-SemiBold.ttf\", preload: false });\n",
            1000,
            500,
        );
        let output = generate(&root, &scan_at(&root.join("app/layout.tsx")), "").unwrap();
        assert!(output.preloads.is_empty(), "{:?}", output.preloads);
        assert_eq!(output.assets.len(), 1);
    }

    /// `declarations` are emitted verbatim, ahead of the generated descriptors, and a
    /// `font-family` among them replaces the generated one — Next's own rule.
    #[test]
    fn declarations_are_emitted_and_can_supply_the_family() {
        let root = scaffold_local(
            "declarations",
            "import localFont from \"next/font/local\";\n\
             const decoFont = localFont({ src: \"../fonts/CalSans-SemiBold.ttf\", \
             declarations: [{ prop: \"font-feature-settings\", value: \"'ss01'\" }] });\n",
            1000,
            500,
        );
        let css = generate(&root, &scan_at(&root.join("app/layout.tsx")), "").unwrap().css;
        assert!(
            css.contains("@font-face { font-feature-settings: 'ss01'; font-family: decoFont; src:"),
            "{css}"
        );
    }

    /// A `declarations` entry naming a descriptor the loader itself writes is an error
    /// under Next, and must be one here rather than silently emitting the property twice.
    #[test]
    fn a_reserved_declaration_is_a_hard_error() {
        let error = scan_next_font(
            Path::new("app/layout.tsx"),
            "import localFont from \"next/font/local\";\n\
             const f = localFont({ src: \"./x.woff2\", declarations: [{ prop: \"src\", value: \"y\" }] });\n",
        )
        .unwrap_err();
        assert!(error.contains("declaration `src`"), "{error}");
        assert!(error.contains("app/layout.tsx"), "{error}");
    }

    /// A `src` that does not exist must name the module, the specifier AND where it
    /// resolved to — the build cannot proceed with a font it cannot read.
    #[test]
    fn a_missing_src_file_is_a_hard_error_naming_the_module_and_the_path() {
        let root = scaffold_local(
            "missing-src",
            "import localFont from \"next/font/local\";\n\
             const goneFont = localFont({ src: \"../fonts/Nope.woff2\" });\n",
            1000,
            500,
        );
        let error = generate(&root, &scan_at(&root.join("app/layout.tsx")), "").unwrap_err();
        assert!(error.contains("layout.tsx"), "{error}");
        assert!(error.contains("../fonts/Nope.woff2"), "{error}");
        assert!(error.contains("fonts/Nope.woff2"), "the resolved path too: {error}");
    }

    /// A file whose metrics cannot be read is a hard error, NOT a silent
    /// `size-adjust: 100%` — the latter is invisible and reintroduces the layout shift
    /// the feature exists to prevent.
    #[test]
    fn an_unreadable_font_file_is_a_hard_error() {
        let root = scaffold_local(
            "corrupt",
            "import localFont from \"next/font/local\";\n\
             const brokenFont = localFont({ src: \"../fonts/CalSans-SemiBold.ttf\" });\n",
            1000,
            500,
        );
        std::fs::write(root.join("fonts/CalSans-SemiBold.ttf"), b"not a font at all").unwrap();
        let error = generate(&root, &scan_at(&root.join("app/layout.tsx")), "").unwrap_err();
        assert!(error.contains("cannot read the metrics"), "{error}");
        assert!(error.contains("CalSans-SemiBold.ttf"), "{error}");
    }

    /// The options Next validates, validated here too — with the reason, not a generic
    /// parse failure.
    #[test]
    fn invalid_local_font_options_are_hard_errors() {
        let missing_src = scan_next_font(
            Path::new("app/layout.tsx"),
            "import localFont from \"next/font/local\";\nconst f = localFont({});\n",
        )
        .unwrap_err();
        assert!(missing_src.contains("missing the required `src`"), "{missing_src}");

        let bad_extension = scan_next_font(
            Path::new("app/layout.tsx"),
            "import localFont from \"next/font/local\";\nconst f = localFont({ src: \"./x.png\" });\n",
        )
        .unwrap_err();
        assert!(bad_extension.contains("./x.png"), "{bad_extension}");
        assert!(bad_extension.contains(".woff2"), "{bad_extension}");

        let bad_display = scan_next_font(
            Path::new("app/layout.tsx"),
            "import localFont from \"next/font/local\";\nconst f = localFont({ src: \"./x.woff2\", display: \"sideways\" });\n",
        )
        .unwrap_err();
        assert!(bad_display.contains("sideways"), "{bad_display}");
        assert!(bad_display.contains("swap"), "{bad_display}");

        // A computed `src` cannot be read at build time; saying so beats emitting a
        // face with no source.
        let computed = scan_next_font(
            Path::new("app/layout.tsx"),
            "import localFont from \"next/font/local\";\nconst p = \"./x.woff2\";\nconst f = localFont({ src: p });\n",
        )
        .unwrap_err();
        assert!(computed.contains("string literal"), "{computed}");
    }

    /// Next derives the family name from the const, so an unassigned call has none —
    /// and inventing one would render the page in a font the app never named.
    #[test]
    fn an_unassigned_local_font_call_is_a_hard_error() {
        let error = scan_next_font(
            Path::new("app/layout.tsx"),
            "import localFont from \"next/font/local\";\nexport default localFont({ src: \"./x.woff2\" });\n",
        )
        .unwrap_err();
        assert!(error.contains("assigned to a variable"), "{error}");
        assert!(error.contains("app/layout.tsx"), "{error}");
    }

    /// Google and local fonts coexist in one module (cal.com's layout does exactly
    /// this): the google `@import` and the local `@font-face` both come out.
    #[test]
    fn a_module_can_mix_google_and_local_fonts() {
        let root = scaffold_local(
            "mixed",
            "import { Inter } from \"next/font/google\";\n\
             import localFont from \"next/font/local\";\n\
             const interFont = Inter({ subsets: [\"latin\"], variable: \"--font-sans\" });\n\
             const calFont = localFont({ src: \"../fonts/CalSans-SemiBold.ttf\", variable: \"--font-cal\" });\n",
            1000,
            500,
        );
        let metrics = root.join("node_modules/next/dist/server");
        std::fs::create_dir_all(&metrics).unwrap();
        std::fs::write(
            metrics.join("capsize-font-metrics.json"),
            r#"{
              "inter": { "category": "sans-serif", "ascent": 1984, "descent": -494,
                         "lineGap": 0, "unitsPerEm": 2048, "xWidthAvg": 978 },
              "arial": { "category": "sans-serif", "ascent": 1854, "descent": -434,
                         "lineGap": 67, "unitsPerEm": 2048, "xWidthAvg": 913 }
            }"#,
        )
        .unwrap();
        let usages = scan_at(&root.join("app/layout.tsx"));
        assert_eq!(usages.len(), 2);
        let output = generate(&root, &usages, "").unwrap();
        assert!(output.css.contains("fonts.googleapis.com/css2?family=Inter"), "{}", output.css);
        assert!(output.css.contains("'Inter Fallback'"), "{}", output.css);
        assert!(output.css.contains("font-family: calFont;"), "{}", output.css);
        assert!(output.css.contains("'calFont Fallback'"), "{}", output.css);
        assert_eq!(output.assets.len(), 1, "only the local font is an asset");
    }

    /// cal.com constructs `calFont` in THREE modules — `app/layout.tsx`,
    /// `app/icons/page.tsx` and `components/PageWrapper.tsx` — with `src` written
    /// relative to each. Identical usages must collapse to one rule (they used to
    /// repeat once per module), while a usage that differs (here: `display`) must still
    /// be emitted, because dropping it would render that module in a face it never
    /// asked for.
    #[test]
    fn repeated_usages_collapse_but_differing_ones_do_not() {
        let root = scaffold_local(
            "repeated",
            "import localFont from \"next/font/local\";\n\
             const calFont = localFont({ src: \"../fonts/CalSans-SemiBold.ttf\", \
             variable: \"--font-cal\", display: \"block\", weight: \"600\" });\n",
            1000,
            500,
        );
        std::fs::create_dir_all(root.join("components")).unwrap();
        // Same font, same options, `src` written relative to a different directory.
        std::fs::write(
            root.join("components/PageWrapper.tsx"),
            "import localFont from \"next/font/local\";\n\
             const calFont = localFont({ src: \"../fonts/CalSans-SemiBold.ttf\", \
             variable: \"--font-cal\", display: \"block\", weight: \"600\" });\n",
        )
        .unwrap();
        // Same font, DIFFERENT display.
        std::fs::create_dir_all(root.join("app/icons")).unwrap();
        std::fs::write(
            root.join("app/icons/page.tsx"),
            "import localFont from \"next/font/local\";\n\
             const calFont = localFont({ src: \"../../fonts/CalSans-SemiBold.ttf\", \
             variable: \"--font-cal\", display: \"swap\", weight: \"600\" });\n",
        )
        .unwrap();
        let mut usages = scan_at(&root.join("app/layout.tsx"));
        usages.extend(scan_at(&root.join("components/PageWrapper.tsx")));
        usages.extend(scan_at(&root.join("app/icons/page.tsx")));
        assert_eq!(usages.len(), 3);

        let output = generate(&root, &usages, "").unwrap();
        let css = &output.css;
        assert_eq!(
            css.matches(".__df_font_calfont {").count(),
            1,
            "the identical class rule must be emitted once: {css}"
        );
        assert_eq!(
            css.matches(".__df_fontvar_calfont {").count(),
            1,
            "and so must the variable rule: {css}"
        );
        assert_eq!(css.matches("font-display: block;").count(), 1, "{css}");
        assert_eq!(css.matches("font-display: swap;").count(), 1, "{css}");
        assert_eq!(
            css.matches("'calFont Fallback'; src: local(\"Arial\")").count(),
            1,
            "one metric-matched face for the family: {css}"
        );
        // One file on disk, so one asset and one preload however many modules use it.
        assert_eq!(output.assets.len(), 1);
        assert_eq!(output.preloads.len(), 1);
    }

    /// A stale manifest must never make the emit copy a font the current CSS does not
    /// reference: every configure pass rewrites it, empty list included.
    #[test]
    fn the_font_manifest_is_rewritten_even_when_empty() {
        let mut root = std::env::temp_dir();
        root.push(format!("diffpack-next-font-manifest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let adapter = root.join(crate::next_adapter::ADAPTER_DIR);
        std::fs::create_dir_all(&adapter).unwrap();
        write_font_manifest(
            &adapter,
            &[FontAsset { source: root.join("nope.woff2"), file: "_diffpack-font/nope.woff2".into() }],
        )
        .unwrap();
        write_font_manifest(&adapter, &[]).unwrap();
        assert_eq!(emit_font_assets(&root, &root.join("out/public")).unwrap(), 0);
    }

    /// Next's `pickFontFileForFallbackGeneration`, including the variable-font range
    /// rule (`"100 900"` contains 400, so its distance is 0).
    #[test]
    fn weight_distance_matches_nexts_rule() {
        assert_eq!(distance_from_normal_weight(None), 0.0);
        assert_eq!(distance_from_normal_weight(Some("400")), 0.0);
        assert_eq!(distance_from_normal_weight(Some("normal")), 0.0);
        assert_eq!(distance_from_normal_weight(Some("bold")), 300.0);
        assert_eq!(distance_from_normal_weight(Some("100 900")), 0.0);
        assert_eq!(distance_from_normal_weight(Some("500 900")), 100.0);
        assert_eq!(distance_from_normal_weight(Some("100 300")), -100.0);
    }

    /// Next keys its metrics table by a camel-cased, space-stripped family name.
    #[test]
    fn metric_keys_match_nexts_format_name() {
        assert_eq!(format_name("Inter"), "inter");
        assert_eq!(format_name("Geist Mono"), "geistMono");
        assert_eq!(format_name("Times New Roman"), "timesNewRoman");
        assert_eq!(format_name("PT Sans"), "pTSans");
    }
}
