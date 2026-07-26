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
use std::path::{Path, PathBuf};

/// The intrinsic MDX elements that a root `mdx-components.tsx` may override. When such a
/// file exists, every emitted element below is rendered through a resolved `_components`
/// map (`_components.h1`, ...) whose defaults are these tag names, so an unspecified tag
/// falls back to the real intrinsic (`"h1"`) and an overridden one uses the app's
/// component. This is exactly the set the emitter can produce.
const INTRINSIC_TAGS: &[&str] = &[
    "h1", "h2", "h3", "h4", "h5", "h6", "p", "a", "blockquote", "ul", "ol", "li", "em",
    "strong", "code", "pre", "img", "hr", "br",
];

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

    // If the app defines a root `mdx-components.tsx` (Next's `useMDXComponents` override
    // convention), every intrinsic element is rendered through the resolved map instead
    // of a plain intrinsic tag. Absence of the file keeps the emitted JSX exactly as
    // before (`use_map` = false).
    let components_file = find_mdx_components(path);
    let use_map = components_file.is_some();

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
            other => emit_node(other, &mut body, path, use_map)?,
        }
    }

    // Emit `export const metadata` from title/description frontmatter so the app-router
    // metadata resolver (which reads named exports at render time) picks it up.
    let mut meta_export = String::new();
    let mut fields = Vec::new();
    if let Some(t) = frontmatter.get("title") {
        fields.push(format!("title: {}", js_string(t)));
    }
    if let Some(d) = frontmatter.get("description") {
        fields.push(format!("description: {}", js_string(d)));
    }
    if !fields.is_empty() {
        meta_export = format!("export const metadata = {{ {} }};\n", fields.join(", "));
    }

    let jsx = if let Some(components_file) = components_file {
        // Import the app's `useMDXComponents`, resolve the override map ONCE per render,
        // and layer it over the intrinsic defaults (so an unspecified tag stays intrinsic)
        // and any `props.components` (MDXProvider nesting). The body already emits every
        // intrinsic as `_components.<tag>`.
        let specifier = js_string(&relative_import_specifier(path, &components_file));
        let defaults = INTRINSIC_TAGS
            .iter()
            .map(|tag| format!("{tag}: {}", js_string(tag)))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "{hoisted}\nimport {{ useMDXComponents as _provideComponents }} from {specifier};\n{meta_export}\
             export default function MDXContent(props) {{\n  \
             const _components = {{ {defaults}, ..._provideComponents(), ...((props && props.components) || {{}}) }};\n  \
             return (<>{body}</>);\n}}\n"
        )
    } else {
        format!(
            "{hoisted}\n{meta_export}export default function MDXContent() {{\n  return (<>{body}</>);\n}}\n"
        )
    };
    Ok(CompiledMdx { jsx, frontmatter })
}

/// Emit one mdast node as JSX into `out`. Unhandled nodes hard-error (naming node+file).
/// When `use_map` is set every intrinsic tag is emitted as `_components.<tag>` (resolved
/// against the app's `mdx-components.tsx`); otherwise the plain intrinsic tag is emitted.
fn emit_node(node: &Node, out: &mut String, path: &Path, use_map: bool) -> Result<(), String> {
    match node {
        Node::Heading(h) => wrap(out, &format!("h{}", h.depth), &h.children, path, use_map)?,
        Node::Paragraph(p) => wrap(out, "p", &p.children, path, use_map)?,
        Node::Emphasis(e) => wrap(out, "em", &e.children, path, use_map)?,
        Node::Strong(s) => wrap(out, "strong", &s.children, path, use_map)?,
        Node::Blockquote(b) => wrap(out, "blockquote", &b.children, path, use_map)?,
        Node::ListItem(li) => wrap(out, "li", &li.children, path, use_map)?,
        Node::List(l) => wrap(out, if l.ordered { "ol" } else { "ul" }, &l.children, path, use_map)?,
        Node::Link(l) => {
            let tag = jsx_tag("a", use_map);
            out.push_str(&format!("<{tag} href={}>", js_string(&l.url)));
            emit_children(&l.children, out, path, use_map)?;
            out.push_str(&format!("</{tag}>"));
        }
        Node::Image(i) => {
            out.push_str(&format!(
                "<{} src={} alt={} />",
                jsx_tag("img", use_map),
                js_string(&i.url),
                js_string(&i.alt)
            ));
        }
        Node::Text(t) => out.push_str(&js_expr_string(&t.value)),
        Node::InlineCode(c) => {
            let tag = jsx_tag("code", use_map);
            out.push_str(&format!("<{tag}>{}</{tag}>", js_expr_string(&c.value)));
        }
        Node::Code(c) => {
            let lang = c.lang.as_deref().unwrap_or("");
            let class = if lang.is_empty() {
                String::new()
            } else {
                format!(" className={}", js_string(&format!("language-{lang}")))
            };
            let pre = jsx_tag("pre", use_map);
            let code = jsx_tag("code", use_map);
            out.push_str(&format!(
                "<{pre}><{code}{class}>{}</{code}></{pre}>",
                js_expr_string(&c.value)
            ));
        }
        Node::ThematicBreak(_) => out.push_str(&format!("<{} />", jsx_tag("hr", use_map))),
        Node::Break(_) => out.push_str(&format!("<{} />", jsx_tag("br", use_map))),
        Node::MdxFlowExpression(e) => out.push_str(&format!("{{{}}}", e.value)),
        Node::MdxTextExpression(e) => out.push_str(&format!("{{{}}}", e.value)),
        Node::MdxJsxFlowElement(el) => emit_jsx_element(el.name.as_deref(), &el.attributes, &el.children, out, path, use_map)?,
        Node::MdxJsxTextElement(el) => emit_jsx_element(el.name.as_deref(), &el.attributes, &el.children, out, path, use_map)?,
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

/// `<tag>children</tag>`, where `tag` is an intrinsic name mapped through `_components`
/// when `use_map` is set.
fn wrap(out: &mut String, tag: &str, children: &[Node], path: &Path, use_map: bool) -> Result<(), String> {
    let tag = jsx_tag(tag, use_map);
    out.push_str(&format!("<{tag}>"));
    emit_children(children, out, path, use_map)?;
    out.push_str(&format!("</{tag}>"));
    Ok(())
}

fn emit_children(children: &[Node], out: &mut String, path: &Path, use_map: bool) -> Result<(), String> {
    for child in children {
        emit_node(child, out, path, use_map)?;
    }
    Ok(())
}

/// The JSX tag for an intrinsic element: the resolved-map member `_components.<tag>` when
/// an `mdx-components.tsx` override is in play, otherwise the plain intrinsic name.
fn jsx_tag(tag: &str, use_map: bool) -> String {
    if use_map {
        format!("_components.{tag}")
    } else {
        tag.to_string()
    }
}

/// Reconstruct an MDX JSX element `<Name attr="x" prop={expr} {...spread}>children</Name>`
/// (a fragment when name is None).
fn emit_jsx_element(
    name: Option<&str>,
    attributes: &[AttributeContent],
    children: &[Node],
    out: &mut String,
    path: &Path,
    use_map: bool,
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
    emit_children(children, out, path, use_map)?;
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

/// Locate the app's `mdx-components.{tsx,ts,jsx,js}` (Next's `useMDXComponents` override
/// file), walking up from the MDX source's directory. Next places it at the project root
/// (next to `app/`) or in `src/`; this also matches an `app/mdx-components.*`. The walk
/// stops at the directory holding `package.json` (the project root) so it never reaches an
/// unrelated file higher in the filesystem. Returns `None` when no such file exists (the
/// no-override path). Paths reaching here are canonicalized absolute paths.
fn find_mdx_components(mdx_path: &Path) -> Option<PathBuf> {
    const EXTS: &[&str] = &["tsx", "ts", "jsx", "js"];
    let mut dir = mdx_path.parent();
    while let Some(current) = dir {
        for ext in EXTS {
            let candidate = current.join(format!("mdx-components.{ext}"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
        // The project root is the nearest ancestor with a package.json; do not walk above
        // it (its mdx-components was already checked in this iteration).
        if current.join("package.json").is_file() {
            return None;
        }
        dir = current.parent();
    }
    None
}

/// Build a relative ESM import specifier from the MDX source file to `target`, dropping the
/// extension (the bundler resolves extensionless relative imports) and forcing a leading
/// `./` so it is never mistaken for a bare package specifier.
fn relative_import_specifier(from_file: &Path, target: &Path) -> String {
    let from_dir = from_file.parent().unwrap_or_else(|| Path::new(""));
    let from: Vec<_> = from_dir.components().collect();
    let to: Vec<_> = target.components().collect();

    let mut shared = 0;
    while shared < from.len() && shared < to.len() && from[shared] == to[shared] {
        shared += 1;
    }

    let mut parts: Vec<String> = Vec::new();
    for _ in shared..from.len() {
        parts.push("..".to_string());
    }
    for component in &to[shared..] {
        parts.push(component.as_os_str().to_string_lossy().into_owned());
    }
    // Drop the extension from the final component (the target file name).
    if let Some(last) = parts.last_mut()
        && let Some(stem) = Path::new(last.as_str()).file_stem().and_then(|s| s.to_str())
    {
        *last = stem.to_string();
    }

    let joined = parts.join("/");
    if joined.starts_with("../") || joined == ".." {
        joined
    } else {
        format!("./{joined}")
    }
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

    /// Create a throwaway project dir (with a package.json root) containing an
    /// `mdx-components.tsx` and a nested `page.mdx`, returning the absolute page path so
    /// `compile` sees the same canonicalized-absolute paths the bundler passes.
    fn scaffold_with_components(components_rel_dir: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let mut root = std::env::temp_dir();
        root.push(format!("diffpack-mdx-{}-{}", std::process::id(), components_rel_dir.replace('/', "_")));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("app/blog")).unwrap();
        std::fs::write(root.join("package.json"), "{}").unwrap();
        let comp_dir = root.join(components_rel_dir);
        std::fs::create_dir_all(&comp_dir).unwrap();
        std::fs::write(
            comp_dir.join("mdx-components.tsx"),
            "export function useMDXComponents() { return {}; }\n",
        )
        .unwrap();
        let page = root.join("app/blog/page.mdx");
        std::fs::write(&page, "# Hi\n").unwrap();
        (root, page)
    }

    #[test]
    fn no_components_file_keeps_plain_intrinsics() {
        // A path with no mdx-components anywhere above it emits plain intrinsic tags and
        // the zero-arg component signature, byte-identical to before this feature.
        let out = jsx("# Title\n\nHi\n");
        assert!(out.contains("<h1>{\"Title\"}</h1>"), "{out}");
        assert!(out.contains("export default function MDXContent() {"), "{out}");
        assert!(!out.contains("_components"), "{out}");
        assert!(!out.contains("_provideComponents"), "{out}");
    }

    #[test]
    fn root_components_file_routes_intrinsics_through_map() {
        let (root, page) = scaffold_with_components(".");
        let out = compile(&page, "# Hi\n\nA [link](/x) and `code`.\n").unwrap().jsx;
        std::fs::remove_dir_all(&root).ok();
        // Imports the app override and resolves the map once.
        assert!(
            out.contains("import { useMDXComponents as _provideComponents } from \"../../mdx-components\""),
            "{out}"
        );
        assert!(out.contains("const _components = {"), "{out}");
        assert!(out.contains("h1: \"h1\""), "{out}");
        assert!(out.contains("..._provideComponents()"), "{out}");
        assert!(out.contains("export default function MDXContent(props)"), "{out}");
        // Every intrinsic is rendered through the map, with the intrinsic fallback baked
        // into `_components`.
        assert!(out.contains("<_components.h1>"), "{out}");
        assert!(out.contains("<_components.a href="), "{out}");
        assert!(out.contains("<_components.code>"), "{out}");
    }

    #[test]
    fn src_app_layout_finds_src_components() {
        // Realistic `src/app` layout: mdx-components lives at the src root, an ancestor of
        // the page, so the walk finds it (and stops at package.json above src/).
        let mut root = std::env::temp_dir();
        root.push(format!("diffpack-mdx-srcapp-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        std::fs::create_dir_all(root.join("src/app/blog")).unwrap();
        std::fs::write(root.join("package.json"), "{}").unwrap();
        std::fs::write(
            root.join("src/mdx-components.tsx"),
            "export function useMDXComponents() { return {}; }\n",
        )
        .unwrap();
        let page = root.join("src/app/blog/page.mdx");
        std::fs::write(&page, "# Hi\n").unwrap();
        let out = compile(&page, "# Hi\n").unwrap().jsx;
        std::fs::remove_dir_all(&root).ok();
        assert!(
            out.contains("from \"../../mdx-components\""),
            "src/ override must be found and imported relative to the page: {out}"
        );
        assert!(out.contains("<_components.h1>"), "{out}");
    }

    #[test]
    fn relative_specifier_strips_extension_and_forces_dot_prefix() {
        let from = Path::new("/proj/app/blog/hello/page.mdx");
        let target = Path::new("/proj/mdx-components.tsx");
        assert_eq!(relative_import_specifier(from, target), "../../../mdx-components");

        let sibling = Path::new("/proj/app/blog/mdx-components.ts");
        assert_eq!(relative_import_specifier(Path::new("/proj/app/blog/page.mdx"), sibling), "./mdx-components");
    }

    #[test]
    fn unsupported_node_is_a_clear_hard_error() {
        // A reference-style link definition is not in the supported subset — a clear
        // error naming the node, never a silent drop.
        let err = compile(Path::new("t.mdx"), "[ref]: https://example.com\n").unwrap_err();
        assert!(err.contains("unsupported node") && err.contains("Definition"), "{err}");
    }
}

