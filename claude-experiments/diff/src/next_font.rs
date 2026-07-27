//! Native `next/font` support — the hard blocker on the stock create-next-app.
//!
//! `next/font/google` and `next/font/local` are **build-time macros**, not runtime
//! modules: the published npm package is a placeholder, and Next's SWC loader
//! REPLACES each `Geist({...})` / `localFont({...})` call with a generated object
//! (`{ className, variable, style }`) plus the font's CSS (`@font-face` / a CSS
//! variable class). Importing the real module and calling it throws. Diffpack does
//! the same rewrite natively on the oxc AST (source-to-source, gated on a cheap
//! `next/font` string check, so non-font modules pay nothing), and generates the
//! companion CSS ([`generate_css`]) which the app-router adapter injects into the
//! document `<head>`.
//!
//! Fidelity: the family's real webfont is loaded via a Google Fonts `@import` (so
//! the browser fetches the actual font), and the call's `variable` option is wired
//! to a CSS-variable class exactly as Next does, so `${font.variable}` on `<html>`
//! defines the custom property the app's CSS reads. Self-hosting the font files
//! (Next's default, to avoid the external request) is a later refinement.
//!
//! The other half of what `next/font` IS — and the entire reason it exists — is the
//! metric-matched fallback face: Next emits a local `@font-face` (`Inter Fallback`,
//! `src: local("Arial")`) carrying `size-adjust` / `ascent-override` /
//! `descent-override` / `line-gap-override` computed from the family's real metrics,
//! so the page does not reflow when the webfont finishes loading. Diffpack computes
//! the same four numbers from the same table Next uses — the `capsize-font-metrics`
//! shipped inside the app's own `next` install — with Next's own arithmetic
//! (`next/dist/server/font-utils.js`), so the generated face is byte-comparable with
//! `next build`'s. A family the table does not know is a HARD ERROR: silently
//! dropping the face is exactly the layout shift `next/font` is there to prevent.

use std::path::{Path, PathBuf};

use oxc_allocator::Allocator;
use oxc_ast::ast::{Argument, Expression, ObjectPropertyKind, PropertyKey, Statement};
use oxc_ast_visit::{walk, Visit};
use oxc_parser::Parser;
use oxc_span::Span;

use crate::server_fn::{apply_edits, quote};

/// A resolved `next/font` usage: the display family (`"Geist Mono"`), the CSS
/// variable name from the call's `variable` option (if any), whether it is a Google
/// font, and the deterministic class names the rewrite and the CSS agree on.
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
    /// The call's `adjustFontFallback` option (default `true`): whether Next generates
    /// the metric-matched `"<family> Fallback"` face.
    pub adjust_fallback: bool,
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

/// One imported font-factory binding: its local name, family, and whether it comes
/// from `next/font/google` (vs `next/font/local`).
struct FontImport {
    local: String,
    family: String,
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
                    let name = spec.local.name.to_string();
                    imports.push(FontImport { local: name.clone(), family: family_display(&name), google });
                }
                // `import localFont from "next/font/local"` — the family is not in
                // the import; use the binding name as a stable label.
                Spec::ImportDefaultSpecifier(spec) => {
                    let name = spec.local.name.to_string();
                    imports.push(FontImport { local: name.clone(), family: family_display(&name), google });
                }
                Spec::ImportNamespaceSpecifier(_) => {}
            }
        }
    }
    (imports, import_spans)
}

/// A matched call `Font({...})`: the whole call span (to replace) and the options
/// the generated CSS and object depend on.
struct MatchedCall {
    span: Span,
    options: FontOptions,
}

/// The subset of a `next/font` call's options that changes what is EMITTED.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct FontOptions {
    /// `variable: "--font-x"`.
    variable: Option<String>,
    /// `fallback: ["Helvetica", "Arial"]`, appended to the family list verbatim.
    fallbacks: Vec<String>,
    /// `adjustFontFallback: false` opts OUT of the metric-matched fallback face.
    adjust_fallback: bool,
}

impl FontOptions {
    /// Next's defaults when the call passes no options at all: the metric-matched
    /// fallback IS generated (`adjustFontFallback` defaults to `true`).
    fn defaults() -> Self {
        Self { variable: None, fallbacks: Vec::new(), adjust_fallback: true }
    }
}

/// Visitor collecting every call whose callee is one of the font bindings.
struct CallCollector<'a> {
    names: &'a [String],
    calls: Vec<(String, MatchedCall)>,
}

impl<'a> Visit<'a> for CallCollector<'a> {
    fn visit_call_expression(&mut self, call: &oxc_ast::ast::CallExpression<'a>) {
        if let Expression::Identifier(ident) = &call.callee
            && self.names.iter().any(|n| n == ident.name.as_str()) {
                let options = call
                    .arguments
                    .first()
                    .map(read_options)
                    .unwrap_or_else(FontOptions::defaults);
                self.calls.push((
                    ident.name.to_string(),
                    MatchedCall { span: call.span, options },
                ));
            }
        walk::walk_call_expression(self, call);
    }
}

/// Reads the emit-relevant options (`variable`, `fallback`, `adjustFontFallback`)
/// out of a font call's first argument.
fn read_options(arg: &Argument) -> FontOptions {
    let mut options = FontOptions::defaults();
    let Argument::ObjectExpression(object) = arg else { return options };
    for property in &object.properties {
        let ObjectPropertyKind::ObjectProperty(prop) = property else { continue };
        let key = match &prop.key {
            PropertyKey::StaticIdentifier(ident) => ident.name.as_str(),
            PropertyKey::StringLiteral(lit) => lit.value.as_str(),
            _ => continue,
        };
        match key {
            "variable" => {
                if let Expression::StringLiteral(value) = &prop.value {
                    options.variable = Some(value.value.to_string());
                }
            }
            "fallback" => {
                if let Expression::ArrayExpression(array) = &prop.value {
                    for element in &array.elements {
                        if let Some(Expression::StringLiteral(value)) = element.as_expression() {
                            options.fallbacks.push(value.value.to_string());
                        }
                    }
                }
            }
            // Only `adjustFontFallback: false` turns the face off; Next treats every
            // other value (including `'Arial'` for next/font/local) as "generate it".
            "adjustFontFallback" => {
                if let Expression::BooleanLiteral(value) = &prop.value {
                    options.adjust_fallback = value.value;
                }
            }
            _ => {}
        }
    }
    options
}

/// The family list a `next/font` object and its CSS class both carry:
/// `'Inter', 'Inter Fallback', <user fallbacks…>` — exactly Next's
/// `postcss-next-font` `formattedFontFamilies`.
fn font_stack(family: &str, options: &FontOptions) -> String {
    let mut families = vec![format!("'{family}'")];
    if options.adjust_fallback {
        families.push(format!("'{family} Fallback'"));
    }
    families.extend(options.fallbacks.iter().cloned());
    families.join(", ")
}

/// The `{ className, variable, style }` literal that replaces a font call, matching
/// what `next/font` produces at build time.
fn font_object(import: &FontImport, options: &FontOptions) -> String {
    let s = slug(&import.local);
    format!(
        "{{ className: {}, variable: {}, style: {{ fontFamily: {} }} }}",
        quote(&format!("__df_font_{s}")),
        quote(&format!("__df_fontvar_{s}")),
        quote(&font_stack(&import.family, options)),
    )
}

/// Rewrites a module's `next/font` calls into static objects and removes the
/// `next/font/*` imports, or `Ok(None)` when the module uses no `next/font`.
///
/// `next/font/local` is a HARD ERROR: its family name, its `@font-face` src and its
/// fallback metrics all come from reading the actual font FILE (Next runs fontkit
/// over it), none of which diffpack does. Rewriting the call anyway would ship a
/// page rendering in a font the app never asked for, silently.
pub fn transform_next_font(path: &Path, source: &str) -> Result<Option<String>, String> {
    if !source.contains("next/font") {
        return Ok(None);
    }
    let allocator = Allocator::default();
    let source_type = crate::parser::scan_source_type(path);
    let parsed = Parser::new(&allocator, source, source_type).parse();
    let program = &parsed.program;

    let (imports, import_spans) = collect_font_imports(program);
    if imports.is_empty() {
        return Ok(None);
    }
    reject_local_fonts(path, &imports)?;
    let names: Vec<String> = imports.iter().map(|i| i.local.clone()).collect();
    let mut collector = CallCollector { names: &names, calls: Vec::new() };
    collector.visit_program(program);

    let mut edits: Vec<(Span, String)> = Vec::new();
    for span in import_spans {
        edits.push((span, String::new()));
    }
    for (name, call) in &collector.calls {
        let Some(import) = imports.iter().find(|i| &i.local == name) else { continue };
        edits.push((call.span, font_object(import, &call.options)));
    }
    Ok(Some(apply_edits(source, String::new(), edits)))
}

/// `next/font/local` is not implemented — say so, naming the module, rather than
/// emitting a font object for a family that does not exist.
fn reject_local_fonts(path: &Path, imports: &[FontImport]) -> Result<(), String> {
    let Some(import) = imports.iter().find(|i| !i.google) else { return Ok(()) };
    Err(format!(
        "{}: `next/font/local` is not implemented (imported as `{}`). Next reads the \
         font FILE to derive its family name, its @font-face src and the metrics of the \
         size-adjusted fallback; diffpack does none of that, and emitting a font object \
         anyway would render the page in a font the app never asked for. Use \
         `next/font/google`, or load the face with your own @font-face rule.",
        path.display(),
        import.local,
    ))
}

/// Scans a module for its `next/font` usages (family + `variable` option +
/// deterministic class names), for the app-router adapter to generate the matching
/// CSS. Mirrors [`transform_next_font`]'s naming so the classes agree.
pub fn scan_next_font(path: &Path, source: &str) -> Result<Vec<FontUsage>, String> {
    if !source.contains("next/font") {
        return Ok(Vec::new());
    }
    let allocator = Allocator::default();
    let source_type = crate::parser::scan_source_type(path);
    let parsed = Parser::new(&allocator, source, source_type).parse();
    let program = &parsed.program;
    let (imports, _) = collect_font_imports(program);
    if imports.is_empty() {
        return Ok(Vec::new());
    }
    reject_local_fonts(path, &imports)?;
    let names: Vec<String> = imports.iter().map(|i| i.local.clone()).collect();
    let mut collector = CallCollector { names: &names, calls: Vec::new() };
    collector.visit_program(program);

    let mut usages = Vec::new();
    for (name, call) in collector.calls {
        let Some(import) = imports.iter().find(|i| i.local == name) else { continue };
        let s = slug(&import.local);
        usages.push(FontUsage {
            family: import.family.clone(),
            variable: call.options.variable,
            class_name: format!("__df_font_{s}"),
            variable_class: format!("__df_fontvar_{s}"),
            google: import.google,
            fallbacks: call.options.fallbacks,
            adjust_fallback: call.options.adjust_fallback,
        });
    }
    Ok(usages)
}

/// Generates the CSS for a set of font usages: one Google Fonts `@import` covering
/// all google families, the metric-matched `"<family> Fallback"` `@font-face` per
/// family, a `.className { font-family }` per font, and a `.variableClass { --var:
/// family }` for each usage that declared a `variable`.
///
/// `root` is the project root: the fallback metrics come from the app's OWN `next`
/// install (`next/dist/server/capsize-font-metrics.json`), which is the same table
/// `next build` reads, so the numbers cannot drift from the oracle. A family that
/// table does not carry is a hard error naming the family and the file.
pub fn generate_css(root: &Path, usages: &[FontUsage]) -> Result<String, String> {
    if usages.is_empty() {
        return Ok(String::new());
    }
    let mut css = String::new();
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

    // The metrics table is read at most once per generation, and only when some usage
    // actually asks for a metric-matched fallback.
    let mut metrics: Option<MetricsTable> = None;
    let mut emitted_faces: Vec<String> = Vec::new();
    for usage in usages {
        if !usage.adjust_fallback || emitted_faces.contains(&usage.family) {
            continue;
        }
        let table = match &metrics {
            Some(table) => table,
            None => metrics.insert(MetricsTable::load(root)?),
        };
        let overrides = table.size_adjust_values(&usage.family)?;
        css.push_str(&overrides.font_face(&usage.family));
        emitted_faces.push(usage.family.clone());
    }

    for usage in usages {
        let options = FontOptions {
            variable: usage.variable.clone(),
            fallbacks: usage.fallbacks.clone(),
            adjust_fallback: usage.adjust_fallback,
        };
        let stack = font_stack(&usage.family, &options);
        css.push_str(&format!(".{} {{ font-family: {stack}; }}\n", usage.class_name));
        if let Some(variable) = &usage.variable {
            css.push_str(&format!(
                ".{} {{ {variable}: {stack}; }}\n",
                usage.variable_class
            ));
        }
    }
    Ok(css)
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

    /// `next/font/local` reads the font FILE (family name, src, fontkit metrics).
    /// Diffpack does not, so it must say so rather than emit an object naming a family
    /// that does not exist.
    #[test]
    fn local_fonts_are_a_loud_error_not_a_wrong_font() {
        let error = transform_next_font(
            Path::new("app/layout.tsx"),
            "import localFont from \"next/font/local\";\nconst f = localFont({ src: \"./x.woff2\" });\n",
        )
        .unwrap_err();
        assert!(error.contains("next/font/local"), "{error}");
        assert!(error.contains("app/layout.tsx"), "{error}");
        assert!(error.contains("not implemented"), "{error}");
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
