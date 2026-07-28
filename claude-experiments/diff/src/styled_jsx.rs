//! Next's styled-jsx JSX compile, minimally and faithfully.
//!
//! Under `next build`, SWC compiles a `<style jsx global>{css}</style>` element
//! into a `<_JSXStyle id={hash}>{css}</_JSXStyle>` referencing the `styled-jsx`
//! package — whose `JSXStyle` component renders **no DOM node at all**: on the
//! client it inserts the stylesheet into the document head from an insertion
//! effect, and on the server (App Router, where no style registry is installed)
//! it does nothing. Without this compile, the raw `<style jsx global>` stays a
//! REAL `<style>` element in the React tree — and cal.com's theme provider
//! renders one behind `typeof window !== "undefined"`, so the client tree held
//! an element the server HTML never had. React reported exactly that hydration
//! mismatch on every route the provider wraps and regenerated the tree on the
//! client (defects #50/#52), which is what made the booker un-automatable under
//! cal.com's own Playwright suite.
//!
//! The compile targets the app's OWN `styled-jsx` install (it ships as a `next`
//! dependency), so client-side insertion behavior is the reference's own code.
//! The SSR no-op also matches the reference: a served `next start` document for
//! cal.com contains zero `__jsx-` style tags.
//!
//! SCOPED styled-jsx (`<style jsx>` without `global`) is a hard error naming
//! the file: its compile rewrites sibling class names, which this transform
//! does not implement — passing it through un-scoped would silently mis-style,
//! and pretending it worked is worse than refusing.
//!
//! The `id` is a hash of the style's SOURCE TEXT, so two renders of one site
//! dedupe in styled-jsx's registry. An interpolated value inside the css
//! (cal.com's `--font-sans: ${interFont.style.fontFamily}`) is evaluated at
//! render time as the element's children, exactly as under Next; the hash does
//! not fold in runtime values, so a site whose interpolations change BETWEEN
//! renders would keep its first inserted text. Next's own transform hashes the
//! static template the same way.

use std::path::Path;

use oxc_allocator::Allocator;
use oxc_ast::ast::{
    JSXAttributeItem, JSXAttributeName, JSXElement, JSXElementName,
};
use oxc_ast_visit::{walk, Visit};
use oxc_parser::Parser;
use oxc_span::Span;

use crate::server_fn::apply_edits;

/// The binding the injected import declares. Prefixed so it can never collide
/// with an app identifier that a bundler user would plausibly write.
const JSX_STYLE_BINDING: &str = "__diffpack_JSXStyle";

/// Rewrites every `<style jsx global>` element to the styled-jsx runtime
/// component. `Ok(None)` = the module has no styled-jsx (the common case, gated
/// on a cheap string check). `Err` = the module uses SCOPED styled-jsx, which
/// is refused rather than mis-compiled.
pub fn transform_styled_jsx(path: &Path, source: &str) -> Result<Option<String>, String> {
    if !source.contains("<style") || !source.contains("jsx") {
        return Ok(None);
    }
    let allocator = Allocator::default();
    let source_type = crate::parser::scan_source_type(path);
    let parsed = Parser::new(&allocator, source, source_type).parse();
    let mut collector = StyleJsxCollector {
        sites: Vec::new(),
        scoped: None,
    };
    collector.visit_program(&parsed.program);
    if let Some(span) = collector.scoped {
        let line = source[..span.start as usize].lines().count();
        return Err(format!(
            "{}:{line}: scoped `<style jsx>` is not supported: its compile rewrites the \
             sibling elements' class names, which diffpack does not implement. Passing it \
             through would apply the styles UNSCOPED (silently wrong), so it is refused. \
             Use `<style jsx global>` with explicit selectors, or a CSS module",
            path.display(),
        ));
    }
    if collector.sites.is_empty() {
        return Ok(None);
    }
    let mut edits: Vec<(Span, String)> = Vec::new();
    for site in &collector.sites {
        let css_source = site
            .children
            .map(|span| &source[span.start as usize..span.end as usize])
            .unwrap_or("");
        let id = format!("df{:016x}", fnv1a_64(css_source.as_bytes()));
        edits.push((
            site.opening,
            format!("<{JSX_STYLE_BINDING} id=\"{id}\">"),
        ));
        match site.closing {
            Some(closing) => edits.push((closing, format!("</{JSX_STYLE_BINDING}>"))),
            None => {
                // Self-closing carries no css at all; still compiled (to nothing
                // rendered) so the element cannot reach the DOM.
                edits.pop();
                edits.push((site.opening, format!("<{JSX_STYLE_BINDING} id=\"{id}\" />")));
            }
        }
    }
    // The import goes AFTER the directive prologue: `"use client"` must stay the
    // module's first statement or the RSC boundary detection downstream would
    // stop seeing it.
    let directive_end = parsed
        .program
        .directives
        .last()
        .map_or(0, |directive| directive.span.end);
    let leading = if directive_end == 0 { "" } else { "\n" };
    edits.push((
        Span::new(directive_end, directive_end),
        format!("{leading}import {JSX_STYLE_BINDING} from \"styled-jsx/style\";\n"),
    ));
    Ok(Some(apply_edits(source, String::new(), edits)))
}

struct StyleJsxSite {
    /// The whole opening element, `<style ... >`.
    opening: Span,
    /// The whole closing element, `</style>`; `None` when self-closing.
    closing: Option<Span>,
    /// The source range between the tags (the css template).
    children: Option<Span>,
}

struct StyleJsxCollector {
    sites: Vec<StyleJsxSite>,
    scoped: Option<Span>,
}

impl<'a> Visit<'a> for StyleJsxCollector {
    fn visit_jsx_element(&mut self, element: &JSXElement<'a>) {
        let is_style = matches!(
            &element.opening_element.name,
            JSXElementName::Identifier(identifier) if identifier.name == "style"
        );
        if is_style {
            let mut has_jsx = false;
            let mut has_global = false;
            for attribute in &element.opening_element.attributes {
                if let JSXAttributeItem::Attribute(attribute) = attribute
                    && let JSXAttributeName::Identifier(name) = &attribute.name
                {
                    match name.name.as_str() {
                        "jsx" => has_jsx = true,
                        "global" => has_global = true,
                        _ => {}
                    }
                }
            }
            if has_jsx && !has_global {
                self.scoped.get_or_insert(element.span);
            }
            if has_jsx && has_global {
                let children = element.closing_element.as_ref().map(|closing| {
                    Span::new(element.opening_element.span.end, closing.span.start)
                });
                self.sites.push(StyleJsxSite {
                    opening: element.opening_element.span,
                    closing: element.closing_element.as_ref().map(|closing| closing.span),
                    children,
                });
            }
        }
        walk::walk_jsx_element(self, element);
    }
}

/// FNV-1a, 64-bit: tiny, dependency-free, deterministic across builds — all the
/// id needs (it is a dedupe key, not a security boundary).
fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tsx() -> PathBuf {
        PathBuf::from("/app/component.tsx")
    }

    #[test]
    fn global_style_jsx_compiles_to_the_styled_jsx_component() {
        let source = "\"use client\";\nexport function Theme() {\n  return (\n    <div>\n      {typeof window !== \"undefined\" && (\n        <style jsx global>{`\n          .dark { color-scheme: dark; }\n        `}</style>\n      )}\n    </div>\n  );\n}\n";
        let output = transform_styled_jsx(&tsx(), source).unwrap().unwrap();
        // The directive stays first; the import lands after it.
        assert!(output.starts_with("\"use client\";\nimport __diffpack_JSXStyle from \"styled-jsx/style\";"), "{output}");
        assert!(output.contains("<__diffpack_JSXStyle id=\"df"), "{output}");
        assert!(output.contains("</__diffpack_JSXStyle>"), "{output}");
        assert!(!output.contains("<style"), "the raw style element must be gone: {output}");
        // The css template survives verbatim as the component's children.
        assert!(output.contains(".dark { color-scheme: dark; }"), "{output}");
    }

    #[test]
    fn interpolated_global_style_keeps_its_expression_children() {
        let source = "export function F(){ return <style jsx global>{`:root { --font: ${font.family}; }`}</style>; }\n";
        let output = transform_styled_jsx(&tsx(), source).unwrap().unwrap();
        assert!(output.contains("${font.family}"), "{output}");
        assert!(output.starts_with("import __diffpack_JSXStyle"), "{output}");
    }

    #[test]
    fn identical_css_hashes_identically_and_different_css_does_not() {
        let a = transform_styled_jsx(&tsx(), "export const A = () => <style jsx global>{`.x{}`}</style>;\n").unwrap().unwrap();
        let b = transform_styled_jsx(&tsx(), "export const B = () => <style jsx global>{`.x{}`}</style>;\n").unwrap().unwrap();
        let c = transform_styled_jsx(&tsx(), "export const C = () => <style jsx global>{`.y{}`}</style>;\n").unwrap().unwrap();
        let id = |s: &str| s.split("id=\"").nth(1).unwrap().split('"').next().unwrap().to_string();
        assert_eq!(id(&a), id(&b));
        assert_ne!(id(&a), id(&c));
    }

    #[test]
    fn scoped_style_jsx_is_a_hard_error_naming_the_file() {
        let source = "export function F(){ return <div><style jsx>{`.x{}`}</style></div>; }\n";
        let error = transform_styled_jsx(&tsx(), source).unwrap_err();
        assert!(error.contains("/app/component.tsx"), "{error}");
        assert!(error.contains("scoped"), "{error}");
    }

    #[test]
    fn a_plain_style_element_is_left_alone() {
        let source = "export function F(){ return <style>{`.x{}`}</style>; }\n// jsx mention so the gate does not skip\n";
        assert!(transform_styled_jsx(&tsx(), source).unwrap().is_none());
    }
}
