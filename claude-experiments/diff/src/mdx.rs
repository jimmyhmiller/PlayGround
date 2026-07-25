//! MDX (`.mdx`) and Markdown (`.md`) as app-router source: a page compiles to a React
//! Server Component. The compile is a NATIVE Rust source-to-source transform (MDX ->
//! JSX) hooked at the single transform choke point, so the emitted JSX then flows
//! through the existing oxc parse + Transformer + RSC pipeline unchanged. No node, no
//! PostCSS-style shell-out: markdown-rs parses MDX to an mdast and this module emits
//! JSX from it.
//!
//! The supported node set is documented below; any mdast node this emitter does not
//! handle is a HARD ERROR naming the node + file (never a silent default), matching the
//! project's stub rule.

use markdown::mdast::{AttributeContent, AttributeValue, Node};
use markdown::{to_mdast, Constructs, MdxSignal, ParseOptions};
use std::collections::BTreeMap;
use std::path::Path;

/// Whether a path is an MDX/Markdown source (`.mdx` or `.md`).
pub fn is_mdx_path(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("mdx" | "md")
    )
}

/// The result of compiling an MDX source: the emitted JSX module + the parsed
/// frontmatter (for page metadata).
#[derive(Debug)]
pub struct CompiledMdx {
    pub jsx: String,
    pub frontmatter: BTreeMap<String, String>,
}

/// The MDX parse options: MDX constructs (JSX + ESM + expressions, raw HTML off) plus
/// frontmatter, which `ParseOptions::mdx()` leaves off. The `mdx_esm_parse` /
/// `mdx_expression_parse` callbacks are REQUIRED for markdown-rs to recognize an
/// `import`/`export` line or a `{expr}` as JS (without them they fall back to prose).
/// We accept any JS as-is (oxc parses/validates it downstream) — always-Ok is the
/// standard trivial parser; the accumulated block is passed to us complete for a
/// single-line construct, which covers app-router MDX pages.
fn parse_options() -> ParseOptions {
    ParseOptions {
        constructs: Constructs {
            frontmatter: true,
            ..Constructs::mdx()
        },
        mdx_esm_parse: Some(Box::new(|_esm: &str| MdxSignal::Ok)),
        mdx_expression_parse: Some(Box::new(|_expr: &str, _kind: &_| MdxSignal::Ok)),
        ..ParseOptions::mdx()
    }
}

/// Parses just the frontmatter of an MDX source into a key -> value map (used by the
/// app-router adapter to derive page metadata). Returns empty on a parse error or no
/// frontmatter (metadata is best-effort; the page still renders).
pub fn frontmatter(source: &str) -> BTreeMap<String, String> {
    match to_mdast(source, &parse_options()) {
        Ok(Node::Root(root)) => root
            .children
            .iter()
            .find_map(|child| match child {
                Node::Yaml(y) => Some(parse_frontmatter_yaml(&y.value)),
                _ => None,
            })
            .unwrap_or_default(),
        _ => BTreeMap::new(),
    }
}

/// Compiles an MDX/Markdown source into a JSX module string exporting a default
/// `MDXContent` Server Component. Hoisted MDX ESM (imports/exports) precede the
/// component so component imports become normal graph edges.
pub fn compile(path: &Path, source: &str) -> Result<CompiledMdx, String> {
    let tree = to_mdast(source, &parse_options())
        .map_err(|e| format!("MDX parse error in {}: {e}", path.display()))?;
    let root = match tree {
        Node::Root(root) => root,
        other => return Err(format!("MDX {}: expected a Root node, got {}", path.display(), node_kind(&other))),
    };

    let mut hoisted = String::new(); // MDX ESM (import/export) lifted to module scope
    let mut frontmatter = BTreeMap::new();
    let mut body = String::new(); // the JSX children of the fragment

    for child in &root.children {
        match child {
            Node::Yaml(y) => frontmatter = parse_frontmatter_yaml(&y.value),
            Node::MdxjsEsm(esm) => {
                hoisted.push_str(&esm.value);
                hoisted.push('\n');
            }
            other => emit_node(other, &mut body, path)?,
        }
    }

    let jsx = format!(
        "{hoisted}\nexport default function MDXContent() {{\n  return (<>{body}</>);\n}}\n"
    );
    Ok(CompiledMdx { jsx, frontmatter })
}

/// Emit one mdast node as JSX into `out`. Unhandled nodes hard-error (naming node+file).
fn emit_node(node: &Node, out: &mut String, path: &Path) -> Result<(), String> {
    match node {
        Node::Heading(h) => wrap(out, &format!("h{}", h.depth), &h.children, path)?,
        Node::Paragraph(p) => wrap(out, "p", &p.children, path)?,
        Node::Emphasis(e) => wrap(out, "em", &e.children, path)?,
        Node::Strong(s) => wrap(out, "strong", &s.children, path)?,
        Node::Blockquote(b) => wrap(out, "blockquote", &b.children, path)?,
        Node::ListItem(li) => wrap(out, "li", &li.children, path)?,
        Node::List(l) => wrap(out, if l.ordered { "ol" } else { "ul" }, &l.children, path)?,
        Node::Link(l) => {
            out.push_str(&format!("<a href={}>", js_string(&l.url)));
            emit_children(&l.children, out, path)?;
            out.push_str("</a>");
        }
        Node::Image(i) => {
            out.push_str(&format!("<img src={} alt={} />", js_string(&i.url), js_string(&i.alt)));
        }
        Node::Text(t) => out.push_str(&js_expr_string(&t.value)),
        Node::InlineCode(c) => out.push_str(&format!("<code>{}</code>", js_expr_string(&c.value))),
        Node::Code(c) => {
            let lang = c.lang.as_deref().unwrap_or("");
            let class = if lang.is_empty() {
                String::new()
            } else {
                format!(" className={}", js_string(&format!("language-{lang}")))
            };
            out.push_str(&format!("<pre><code{class}>{}</code></pre>", js_expr_string(&c.value)));
        }
        Node::ThematicBreak(_) => out.push_str("<hr />"),
        Node::Break(_) => out.push_str("<br />"),
        Node::MdxFlowExpression(e) => out.push_str(&format!("{{{}}}", e.value)),
        Node::MdxTextExpression(e) => out.push_str(&format!("{{{}}}", e.value)),
        Node::MdxJsxFlowElement(el) => emit_jsx_element(el.name.as_deref(), &el.attributes, &el.children, out, path)?,
        Node::MdxJsxTextElement(el) => emit_jsx_element(el.name.as_deref(), &el.attributes, &el.children, out, path)?,
        // MDX ESM only appears at the top level (hoisted in `compile`); a nested one is
        // malformed. Everything else is an explicitly unsupported node.
        other => {
            return Err(format!(
                "MDX {}: unsupported node `{}` (diffpack's MDX emitter handles headings, paragraphs, \
                 emphasis/strong, links, images, lists, blockquotes, inline/fenced code, thematic breaks, \
                 MDX expressions, and MDX JSX elements)",
                path.display(),
                node_kind(other),
            ));
        }
    }
    Ok(())
}

/// `<tag>children</tag>`.
fn wrap(out: &mut String, tag: &str, children: &[Node], path: &Path) -> Result<(), String> {
    out.push_str(&format!("<{tag}>"));
    emit_children(children, out, path)?;
    out.push_str(&format!("</{tag}>"));
    Ok(())
}

fn emit_children(children: &[Node], out: &mut String, path: &Path) -> Result<(), String> {
    for child in children {
        emit_node(child, out, path)?;
    }
    Ok(())
}

/// Reconstruct an MDX JSX element `<Name attr="x" prop={expr} {...spread}>children</Name>`
/// (a fragment when name is None).
fn emit_jsx_element(
    name: Option<&str>,
    attributes: &[AttributeContent],
    children: &[Node],
    out: &mut String,
    path: &Path,
) -> Result<(), String> {
    let tag = name.unwrap_or("");
    out.push('<');
    out.push_str(tag);
    for attr in attributes {
        match attr {
            AttributeContent::Property(p) => match &p.value {
                None => out.push_str(&format!(" {}", p.name)),
                Some(AttributeValue::Literal(lit)) => out.push_str(&format!(" {}={}", p.name, js_string(lit))),
                Some(AttributeValue::Expression(e)) => out.push_str(&format!(" {}={{{}}}", p.name, e.value)),
            },
            AttributeContent::Expression(e) => out.push_str(&format!(" {{{}}}", e.value)),
        }
    }
    if children.is_empty() {
        out.push_str(" />");
        return Ok(());
    }
    out.push('>');
    emit_children(children, out, path)?;
    out.push_str(&format!("</{tag}>"));
    Ok(())
}

/// A JS string literal `"..."` (for JSX attribute values / urls).
fn js_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Text emitted as a JS-string JSX child `{"..."}`, so no JSX-level escaping of
/// `<`/`{`/`&` is needed (React renders the string verbatim as a text node).
fn js_expr_string(s: &str) -> String {
    format!("{{{}}}", js_string(s))
}

/// Parses simple `key: value` frontmatter lines (quotes stripped). Not a full YAML
/// parser: enough for `title`/`description`, the metadata diffpack derives.
fn parse_frontmatter_yaml(yaml: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for line in yaml.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once(':') {
            let v = v.trim().trim_matches(['"', '\'']).to_string();
            out.insert(k.trim().to_string(), v);
        }
    }
    out
}

/// A human-readable node kind for error messages.
fn node_kind(node: &Node) -> &'static str {
    match node {
        Node::Root(_) => "Root",
        Node::Html(_) => "Html (raw HTML — write it as JSX in MDX)",
        Node::Definition(_) => "Definition (reference-style link defs are unsupported)",
        Node::FootnoteDefinition(_) => "FootnoteDefinition",
        Node::FootnoteReference(_) => "FootnoteReference",
        Node::Table(_) => "Table (GFM tables are unsupported)",
        Node::Math(_) => "Math",
        Node::InlineMath(_) => "InlineMath",
        Node::Delete(_) => "Delete (strikethrough)",
        Node::Yaml(_) => "Yaml",
        Node::Toml(_) => "Toml",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jsx(src: &str) -> String {
        compile(Path::new("page.mdx"), src).unwrap().jsx
    }

    #[test]
    fn detects_mdx_paths() {
        assert!(is_mdx_path(Path::new("a/page.mdx")));
        assert!(is_mdx_path(Path::new("a/readme.md")));
        assert!(!is_mdx_path(Path::new("a/page.tsx")));
    }

    #[test]
    fn frontmatter_becomes_metadata_not_body() {
        let out = compile(Path::new("p.mdx"), "---\ntitle: Hi\ndescription: D\n---\n\n# H\n").unwrap();
        assert_eq!(out.frontmatter.get("title").map(String::as_str), Some("Hi"));
        assert_eq!(out.frontmatter.get("description").map(String::as_str), Some("D"));
        assert!(!out.jsx.contains("title: Hi"), "frontmatter is not in the body: {}", out.jsx);
        assert!(out.jsx.contains("<h1>"), "{}", out.jsx);
    }

    #[test]
    fn headings_paragraphs_and_inline_marks() {
        let out = jsx("# Title\n\nHello **bold** and *em*.\n");
        assert!(out.contains("<h1>{\"Title\"}</h1>"), "{out}");
        assert!(out.contains("<strong>{\"bold\"}</strong>"), "{out}");
        assert!(out.contains("<em>{\"em\"}</em>"), "{out}");
    }

    #[test]
    fn fenced_code_is_escaped_with_language_class() {
        let out = jsx("```js\nconst a = 1 < 2;\n```\n");
        assert!(out.contains("className=\"language-js\""), "{out}");
        // Content is a JS string child (so `<`/`{` need no JSX escaping) and byte-preserved.
        assert!(out.contains("{\"const a = 1 < 2;\"}"), "{out}");
    }

    #[test]
    fn component_import_is_hoisted_and_used() {
        let out = jsx("import Widget from \"./w\";\n\n<Widget n={2} label=\"hi\" />\n");
        // Import hoisted ABOVE the component (module scope), not rendered as prose.
        let import_at = out.find("import Widget").unwrap();
        let component_at = out.find("export default function MDXContent").unwrap();
        assert!(import_at < component_at, "import must be hoisted: {out}");
        assert!(out.contains("<Widget n={2} label=\"hi\" />"), "{out}");
    }

    #[test]
    fn unsupported_node_is_a_clear_hard_error() {
        // A reference-style link definition is not in the supported subset — a clear
        // error naming the node, never a silent drop.
        let err = compile(Path::new("t.mdx"), "[ref]: https://example.com\n").unwrap_err();
        assert!(err.contains("unsupported node") && err.contains("Definition"), "{err}");
    }
}

