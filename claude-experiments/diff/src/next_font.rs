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

use std::path::Path;

use oxc_allocator::Allocator;
use oxc_ast::ast::{Argument, Expression, ObjectPropertyKind, PropertyKey, Statement};
use oxc_ast_visit::{walk, Visit};
use oxc_parser::Parser;
use oxc_span::{SourceType, Span};

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
}

const FALLBACK_STACK: &str = "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif";

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

/// A matched call `Font({...})`: the whole call span (to replace) and the string
/// value of its `variable` option, if a plain string literal.
struct MatchedCall {
    span: Span,
    variable: Option<String>,
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
                let variable = call.arguments.first().and_then(option_variable);
                self.calls.push((
                    ident.name.to_string(),
                    MatchedCall { span: call.span, variable },
                ));
            }
        walk::walk_call_expression(self, call);
    }
}

/// Reads the `variable: "--x"` string option out of a font call's first argument.
fn option_variable(arg: &Argument) -> Option<String> {
    let Argument::ObjectExpression(object) = arg else { return None };
    for property in &object.properties {
        let ObjectPropertyKind::ObjectProperty(prop) = property else { continue };
        let key = match &prop.key {
            PropertyKey::StaticIdentifier(ident) => ident.name.as_str(),
            PropertyKey::StringLiteral(lit) => lit.value.as_str(),
            _ => continue,
        };
        if key == "variable"
            && let Expression::StringLiteral(value) = &prop.value {
                return Some(value.value.to_string());
            }
    }
    None
}

/// The `{ className, variable, style }` literal that replaces a font call, matching
/// what `next/font` produces at build time.
fn font_object(import: &FontImport) -> String {
    let s = slug(&import.local);
    format!(
        "{{ className: {}, variable: {}, style: {{ fontFamily: {} }} }}",
        quote(&format!("__df_font_{s}")),
        quote(&format!("__df_fontvar_{s}")),
        quote(&format!("'{}', {FALLBACK_STACK}", import.family)),
    )
}

/// Rewrites a module's `next/font` calls into static objects and removes the
/// `next/font/*` imports, or `None` when the module uses no `next/font`.
pub fn transform_next_font(path: &Path, source: &str) -> Option<String> {
    if !source.contains("next/font") {
        return None;
    }
    let allocator = Allocator::default();
    let source_type = SourceType::from_path(path).unwrap_or_default().with_module(true);
    let parsed = Parser::new(&allocator, source, source_type).parse();
    let program = &parsed.program;

    let (imports, import_spans) = collect_font_imports(program);
    if imports.is_empty() {
        return None;
    }
    let names: Vec<String> = imports.iter().map(|i| i.local.clone()).collect();
    let mut collector = CallCollector { names: &names, calls: Vec::new() };
    collector.visit_program(program);

    let mut edits: Vec<(Span, String)> = Vec::new();
    for span in import_spans {
        edits.push((span, String::new()));
    }
    for (name, call) in &collector.calls {
        let Some(import) = imports.iter().find(|i| &i.local == name) else { continue };
        edits.push((call.span, font_object(import)));
    }
    Some(apply_edits(source, String::new(), edits))
}

/// Scans a module for its `next/font` usages (family + `variable` option +
/// deterministic class names), for the app-router adapter to generate the matching
/// CSS. Mirrors [`transform_next_font`]'s naming so the classes agree.
pub fn scan_next_font(path: &Path, source: &str) -> Vec<FontUsage> {
    if !source.contains("next/font") {
        return Vec::new();
    }
    let allocator = Allocator::default();
    let source_type = SourceType::from_path(path).unwrap_or_default().with_module(true);
    let parsed = Parser::new(&allocator, source, source_type).parse();
    let program = &parsed.program;
    let (imports, _) = collect_font_imports(program);
    if imports.is_empty() {
        return Vec::new();
    }
    let names: Vec<String> = imports.iter().map(|i| i.local.clone()).collect();
    let mut collector = CallCollector { names: &names, calls: Vec::new() };
    collector.visit_program(program);

    let mut usages = Vec::new();
    for (name, call) in collector.calls {
        let Some(import) = imports.iter().find(|i| i.local == name) else { continue };
        let s = slug(&import.local);
        usages.push(FontUsage {
            family: import.family.clone(),
            variable: call.variable,
            class_name: format!("__df_font_{s}"),
            variable_class: format!("__df_fontvar_{s}"),
            google: import.google,
        });
    }
    usages
}

/// Generates the CSS for a set of font usages: one Google Fonts `@import` covering
/// all google families, a `.className { font-family }` per font, and a
/// `.variableClass { --var: family }` for each usage that declared a `variable`.
pub fn generate_css(usages: &[FontUsage]) -> String {
    if usages.is_empty() {
        return String::new();
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
    for usage in usages {
        css.push_str(&format!(
            ".{} {{ font-family: '{}', {FALLBACK_STACK}; }}\n",
            usage.class_name, usage.family
        ));
        if let Some(variable) = &usage.variable {
            css.push_str(&format!(
                ".{} {{ {}: '{}', {FALLBACK_STACK}; }}\n",
                usage.variable_class, variable, usage.family
            ));
        }
    }
    css
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(source: &str) -> Option<String> {
        transform_next_font(Path::new("app/layout.tsx"), source)
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
        let usages = scan_next_font(
            Path::new("app/layout.tsx"),
            "import { Geist, Geist_Mono } from \"next/font/google\";\nconst a = Geist({ variable: \"--font-geist-sans\" });\nconst b = Geist_Mono({ variable: \"--font-geist-mono\" });\n",
        );
        assert_eq!(usages.len(), 2);
        let css = generate_css(&usages);
        assert!(css.contains("fonts.googleapis.com/css2?family=Geist:wght"), "{css}");
        assert!(css.contains("family=Geist+Mono:wght"), "{css}");
        assert!(css.contains(".__df_fontvar_geist { --font-geist-sans: 'Geist'"), "{css}");
        assert!(css.contains(".__df_font_geist { font-family: 'Geist'"), "{css}");
    }
}
