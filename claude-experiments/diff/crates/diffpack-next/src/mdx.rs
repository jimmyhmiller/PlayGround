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
use markdown::{Constructs, MdxSignal, ParseOptions, to_mdast};
use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock};

/// The intrinsic MDX elements that a root `mdx-components.tsx` may override. When such a
/// file exists, every emitted element below is rendered through a resolved `_components`
/// map (`_components.h1`, ...) whose defaults are these tag names, so an unspecified tag
/// falls back to the real intrinsic (`"h1"`) and an overridden one uses the app's
/// component. This is exactly the set the emitter can produce.
const INTRINSIC_TAGS: &[&str] = &[
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "a",
    "blockquote",
    "ul",
    "ol",
    "li",
    "em",
    "strong",
    "code",
    "pre",
    "img",
    "hr",
    "br",
    // GFM (`remark-gfm`): the extra elements tables, strikethrough, task lists and
    // footnotes can produce. Listed unconditionally so the defaults map has the same shape
    // whether or not a given page uses them — an unused key costs nothing and an app that
    // overrides `table` gets it honoured the moment it writes one.
    "del",
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
    "input",
    "section",
    "sup",
];

/// `mdast-util-to-hast`'s `clobberPrefix` default: every footnote id/href is prefixed with
/// it so a footnote named `title` cannot clobber `document.title` via DOM id lookup.
const CLOBBER_PREFIX: &str = "user-content-";

/// The whitespace text node `mdast-util-to-hast` puts BETWEEN block children (its `wrap()`),
/// emitted the way `hast-util-to-estree` emits it: an explicit `{"\n"}` expression child, not
/// literal JSX whitespace (JSX strips whitespace runs that contain a newline, so a literal
/// one would vanish).
///
/// It is not cosmetic. Two adjacent flow-level *inline* elements — `<Badge/>` followed by
/// `<Counter/>`, which is ordinary MDX — render as `…route clicked 0 times` without it and
/// `…route⎵clicked 0 times` with it, a difference the e2e text channel sees and a reader
/// sees. Where the neighbours are block-level the browser collapses it, which is why its
/// absence went unnoticed for so long.
const EOL: &str = "{\"\\n\"}";

/// `mdast-util-to-hast`'s `listItem` rule for where those newlines go: one before every
/// child EXCEPT a tight list item's leading paragraph (which is unwrapped into the `<li>`).
fn eol_before_list_item_child(loose: bool, index: usize, is_paragraph: bool) -> bool {
    loose || index != 0 || !is_paragraph
}

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

/// Whether GitHub-Flavoured Markdown constructs are enabled for a compile. GFM is NOT
/// CommonMark and `@next/mdx` does not turn it on either: an app opts in by configuring
/// `remark-gfm`, and diffpack honours exactly that signal (see [`MdxConfig::wants_gfm`]).
/// Enabling it unconditionally would render tables and `~~text~~` that the app's own
/// toolchain leaves as literal prose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gfm {
    On,
    Off,
}

impl Gfm {
    fn on(self) -> bool {
        self == Gfm::On
    }
}

/// The MDX parse options: MDX constructs (JSX + ESM + expressions, raw HTML off) plus
/// frontmatter, which `ParseOptions::mdx()` leaves off, plus the six GFM constructs when
/// the app asked for them. The `mdx_esm_parse` / `mdx_expression_parse` callbacks are
/// REQUIRED for markdown-rs to recognize an `import`/`export` line or a `{expr}` as JS
/// (without them they fall back to prose). We accept any JS as-is (oxc parses/validates it
/// downstream) — always-Ok is the standard trivial parser; the accumulated block is passed
/// to us complete for a single-line construct, which covers app-router MDX pages.
fn parse_options(gfm: Gfm) -> ParseOptions {
    let gfm = gfm.on();
    ParseOptions {
        constructs: Constructs {
            frontmatter: true,
            gfm_autolink_literal: gfm,
            gfm_footnote_definition: gfm,
            gfm_label_start_footnote: gfm,
            gfm_strikethrough: gfm,
            gfm_table: gfm,
            gfm_task_list_item: gfm,
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
    match to_mdast(source, &parse_options(Gfm::Off)) {
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

// ---------------------------------------------------------------------------------------
// `@next/mdx` configuration: `createMDX({ options: { remarkPlugins, rehypePlugins, ... } })`
// ---------------------------------------------------------------------------------------

/// The node script that compiles one MDX file with the app's own `@mdx-js/mdx` + plugins.
const RUNNER: &str = include_str!("mdx/runner.mjs");

/// One configured remark/rehype/recma plugin, as reported by `scripts/rsc/next-config-eval.mjs`.
/// Only the plugin's IDENTITY survives that JSON boundary — a plugin value is a live JS
/// function — which is enough to name it in a build error; running it means re-evaluating
/// `next.config` inside [`RUNNER`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MdxPlugin {
    /// `remark-gfm` (a specifier) or `remarkGfm` (a function's `name`).
    pub name: String,
    /// `"specifier"`, `"function"`, or `"unknown"`.
    pub kind: String,
    /// Whether it was configured as `[plugin, options]`.
    pub has_options: bool,
}

impl std::fmt::Display for MdxPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)?;
        if self.has_options {
            write!(f, " (with options)")?;
        }
        Ok(())
    }
}

/// The `@next/mdx` configuration an app declares, read out of the next.config eval.
///
/// `extension` and `mdxRs` are recorded but NOT treated as gaps: `extension` only selects
/// which files are MDX (diffpack's own routing decides that, and it compiles both `.md` and
/// `.mdx`), and `mdxRs` asks Next for a Rust MDX compiler — which is precisely what
/// diffpack's native emitter is.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MdxConfig {
    /// Whether the app wraps its config with `@next/mdx` at all.
    pub configured: bool,
    pub remark_plugins: Vec<MdxPlugin>,
    pub rehype_plugins: Vec<MdxPlugin>,
    pub recma_plugins: Vec<MdxPlugin>,
    pub provider_import_source: Option<String>,
    pub extension: Option<String>,
    pub mdx_rs: bool,
    /// Every other key passed to `createMDX`'s options that diffpack does not model.
    pub other_options: Vec<String>,
}

fn plugins_from_json(value: Option<&serde_json::Value>) -> Vec<MdxPlugin> {
    value
        .and_then(|v| v.as_array())
        .map(|items| {
            items
                .iter()
                .map(|item| MdxPlugin {
                    name: item
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("(unnamed)")
                        .to_string(),
                    kind: item
                        .get("kind")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string(),
                    has_options: item
                        .get("hasOptions")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Whether a configured remark plugin is `remark-gfm` in a form diffpack implements
/// NATIVELY (see [`Gfm`]): the bare specifier `remark-gfm`, or the plugin function's own
/// name (`remarkGfm`, or `gfm` when re-exported), and with NO options object.
///
/// The options exclusion is deliberate. Only a plugin's identity survives the next.config
/// eval's JSON boundary, so `[remarkGfm, { singleTilde: false }]` would arrive here
/// indistinguishable from `remarkGfm` — and `singleTilde` changes what actually parses.
/// An options-carrying gfm therefore stays "unhonored" and the file is compiled by the
/// app's own pipeline, which has the real options object in hand.
fn is_native_gfm(plugin: &MdxPlugin) -> bool {
    !plugin.has_options && matches!(plugin.name.trim(), "remark-gfm" | "remarkGfm" | "gfm")
}

impl MdxConfig {
    /// Whether the app opted into GitHub-Flavoured Markdown by configuring `remark-gfm`
    /// (in the natively-implementable form — see [`is_native_gfm`]). This is the ONLY
    /// signal that turns GFM on; an app that does not configure it gets CommonMark, which
    /// is exactly what `@next/mdx` gives it.
    pub fn wants_gfm(&self) -> bool {
        self.remark_plugins.iter().any(is_native_gfm)
    }

    /// Read the `mdx` block of a next.config eval result. An eval without one (no config,
    /// an older manifest) is simply "not configured".
    pub fn from_eval(eval: Option<&serde_json::Value>) -> MdxConfig {
        let Some(mdx) = eval.and_then(|v| v.get("mdx")) else {
            return MdxConfig::default();
        };
        MdxConfig {
            configured: mdx
                .get("configured")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            remark_plugins: plugins_from_json(mdx.get("remarkPlugins")),
            rehype_plugins: plugins_from_json(mdx.get("rehypePlugins")),
            recma_plugins: plugins_from_json(mdx.get("recmaPlugins")),
            provider_import_source: mdx
                .get("providerImportSource")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            extension: mdx
                .get("extension")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            mdx_rs: mdx.get("mdxRs").and_then(|v| v.as_bool()).unwrap_or(false),
            other_options: mdx
                .get("otherOptions")
                .and_then(|v| v.as_array())
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|v| v.as_str())
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default(),
        }
    }

    /// Everything the app configured that the NATIVE emitter cannot honour, described for a
    /// human. Empty = the native compiler is a faithful implementation of this config.
    /// Non-empty = the app's own pipeline must run (or the build must fail saying why).
    pub fn unhonored_options(&self) -> Vec<String> {
        let mut out = Vec::new();
        let mut list = |label: &str, plugins: &[MdxPlugin]| {
            if !plugins.is_empty() {
                out.push(format!(
                    "{label}: [{}]",
                    plugins
                        .iter()
                        .map(MdxPlugin::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        };
        // `remark-gfm` is filtered out: the native emitter implements GFM (tables,
        // strikethrough, task lists, autolink literals, footnotes) directly, so configuring
        // it is honoured rather than deferred to a node process per file.
        let native_remark: Vec<MdxPlugin> = self
            .remark_plugins
            .iter()
            .filter(|p| !is_native_gfm(p))
            .cloned()
            .collect();
        list("remarkPlugins", &native_remark);
        list("rehypePlugins", &self.rehype_plugins);
        list("recmaPlugins", &self.recma_plugins);
        if let Some(provider) = &self.provider_import_source {
            out.push(format!("providerImportSource: {provider:?}"));
        }
        for key in &self.other_options {
            out.push(format!("options.{key}"));
        }
        out
    }

    /// A one-line summary of the whole `createMDX` configuration, for the build log — so an
    /// author can always see that diffpack READ their options, honoured or not.
    pub fn summary(&self) -> String {
        if !self.configured {
            return "not configured".to_string();
        }
        let mut parts = Vec::new();
        if let Some(extension) = &self.extension {
            parts.push(format!("extension {extension}"));
        }
        if self.mdx_rs {
            parts.push("experimental.mdxRs".to_string());
        }
        if self.wants_gfm() {
            parts.push("remark-gfm (native GFM)".to_string());
        }
        let unhonored = self.unhonored_options();
        if unhonored.is_empty() {
            if !self.wants_gfm() {
                parts.push("no remark/rehype/recma plugins".to_string());
            }
        } else {
            parts.extend(unhonored);
        }
        parts.join("; ")
    }
}

/// A Next project whose MDX configuration has been resolved: where its root is, which
/// `next.config` said so, and what that config asked for.
#[derive(Debug, Clone)]
pub struct MdxProject {
    pub root: PathBuf,
    pub config_path: PathBuf,
    pub config: MdxConfig,
}

/// The nearest ancestor directory of `path` holding a `package.json` — the same project-root
/// rule [`find_mdx_components`] uses.
fn find_project_root(path: &Path) -> Option<PathBuf> {
    let mut dir = path.parent();
    while let Some(current) = dir {
        if current.join("package.json").is_file() {
            return Some(current.to_path_buf());
        }
        dir = current.parent();
    }
    None
}

/// The resolved `@next/mdx` configuration governing `mdx_path`, or None when the file is not
/// inside a Next project with a `next.config` (a hermetic fixture, a Vite app, ...).
///
/// The next.config eval spawns node, so the result is cached: a build compiling fifty MDX
/// pages evaluates the config ONCE. The cache key carries the config file's mtime, so
/// adding a `remarkPlugins` entry under a running dev server takes effect on the next
/// compile rather than being served stale until a restart.
fn project_mdx_config(mdx_path: &Path) -> Option<MdxProject> {
    /// Project root + the mtime of its `next.config.*` (None when unreadable).
    type ConfigKey = (PathBuf, Option<std::time::SystemTime>);
    static CACHE: OnceLock<Mutex<HashMap<ConfigKey, MdxProject>>> = OnceLock::new();
    let root = find_project_root(mdx_path)?;
    let config_path = crate::next_config::next_config_path(&root)?;
    let stamp = std::fs::metadata(&config_path)
        .ok()
        .and_then(|meta| meta.modified().ok());
    let key = (root.clone(), stamp);
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(map) = cache.lock()
        && let Some(hit) = map.get(&key)
    {
        return Some(hit.clone());
    }
    let eval = crate::next_config::run_next_config_eval(&root);
    let resolved = MdxProject {
        root,
        config_path,
        config: MdxConfig::from_eval(eval.as_ref()),
    };
    if let Ok(mut map) = cache.lock() {
        map.insert(key, resolved.clone());
    }
    Some(resolved)
}

/// Compile `source` with the APP's own MDX pipeline (its `@mdx-js/mdx` + the exact plugins
/// its `next.config` configures), by shelling to [`RUNNER`] with cwd = the project root.
///
/// Every failure — no node, no `@mdx-js/mdx`, a plugin that throws — is a hard error that
/// names the file AND the configured plugins, because the alternative (compiling without
/// them) renders a page the author did not write.
fn compile_with_app_pipeline(
    path: &Path,
    source: &str,
    project: &MdxProject,
) -> Result<CompiledMdx, String> {
    let configured = project.config.unhonored_options().join("; ");
    let context = format!(
        "MDX {}: {} configures [{configured}], which diffpack's native MDX compiler cannot \
         run, so the file is compiled with the app's own MDX pipeline",
        path.display(),
        project.config_path.display(),
    );

    let provider = find_mdx_components(path).unwrap_or_default();
    let loader = std::env::temp_dir().join("diffpack-mdx-runner.mjs");
    std::fs::write(&loader, RUNNER)
        .map_err(|error| format!("{context} — cannot write {}: {error}", loader.display()))?;

    let mut child = Command::new("node")
        .arg(&loader)
        .arg(&project.config_path)
        .env("DIFFPACK_MDX_FILE", path)
        .env("DIFFPACK_MDX_PROVIDER", provider)
        .current_dir(&project.root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("{context} — cannot run node: {error}"))?;
    child
        .stdin
        .take()
        .ok_or_else(|| format!("{context} — node stdin unavailable"))?
        .write_all(source.as_bytes())
        .map_err(|error| format!("{context} — cannot write to node: {error}"))?;
    let out = child
        .wait_with_output()
        .map_err(|error| format!("{context} — node failed: {error}"))?;
    if !out.status.success() {
        return Err(format!(
            "{context} — that pipeline failed:\n{}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let parsed: serde_json::Value = serde_json::from_slice(&out.stdout).map_err(|error| {
        format!(
            "{context} — unreadable compiler output ({error}): {}",
            String::from_utf8_lossy(&out.stdout).trim()
        )
    })?;
    let jsx = parsed
        .get("jsx")
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("{context} — compiler output has no `jsx` field"))?;
    Ok(CompiledMdx {
        jsx: jsx.to_string(),
        // Frontmatter is still parsed natively so page metadata is derived the same way for
        // both compilers. Whether the YAML is STRIPPED from the rendered output is the app's
        // pipeline's business here (that is what `remark-frontmatter` is for), exactly as
        // under `next build`.
        frontmatter: frontmatter(source),
    })
}

/// Compiles an MDX/Markdown source into a JSX module string exporting a default
/// `MDXContent` Server Component.
///
/// Two compilers, chosen by what the app's `next.config` asked for:
///
/// * Nothing configured beyond `@next/mdx`'s defaults — the NATIVE Rust emitter below
///   (no node process, hoisted MDX ESM, `mdx-components.*` overrides), in CommonMark mode.
/// * `remarkPlugins: [remarkGfm]` and nothing else — the native emitter with [`Gfm::On`],
///   which implements tables, strikethrough, task lists, autolink literals and footnotes
///   the way `remark-gfm` + `mdast-util-to-hast` do.
/// * Any other `createMDX({ options: { remarkPlugins, rehypePlugins, ... } })` — the APP's
///   OWN pipeline, via [`compile_with_app_pipeline`]. A unified plugin is an
///   arbitrary JS function over an mdast/hast; the native emitter cannot run one, and
///   compiling without it would silently render a different page than the author wrote.
///   When the app's pipeline is unavailable this is a HARD ERROR naming the plugins and
///   the file — never a quiet downgrade to plain CommonMark.
pub fn compile(path: &Path, source: &str) -> Result<CompiledMdx, String> {
    let project = project_mdx_config(path);
    let gfm = match project.as_ref().map(|p| p.config.wants_gfm()) {
        Some(true) => Gfm::On,
        _ => Gfm::Off,
    };
    if let Some(project) = &project
        && !project.config.unhonored_options().is_empty()
    {
        return compile_with_app_pipeline(path, source, project);
    }
    compile_native(path, source, gfm)
}

/// The native markdown-rs -> JSX emitter. See [`compile`] for when it is used.
fn compile_native(path: &Path, source: &str, gfm: Gfm) -> Result<CompiledMdx, String> {
    let tree = to_mdast(source, &parse_options(gfm))
        .map_err(|e| format!("MDX parse error in {}: {e}", path.display()))?;
    let mut root = match tree {
        Node::Root(root) => root,
        other => {
            return Err(format!(
                "MDX {}: expected a Root node, got {}",
                path.display(),
                node_kind(&other)
            ));
        }
    };
    unravel(&mut root.children);

    // If the app defines a root `mdx-components.tsx` (Next's `useMDXComponents` override
    // convention), every intrinsic element is rendered through the resolved map instead
    // of a plain intrinsic tag. Absence of the file keeps the emitted JSX exactly as
    // before (`use_map` = false).
    let components_file = find_mdx_components(path);
    let use_map = components_file.is_some();

    let mut hoisted = String::new(); // MDX ESM (import/export) lifted to module scope
    let mut frontmatter = BTreeMap::new();
    let mut body = String::new(); // the JSX children of the fragment

    let mut emitter = Emitter::new(path, use_map, &root.children);
    // `mdast-util-to-hast`'s root handler is `wrap(children)`: a `{"\n"}` BETWEEN every pair
    // of emitted root children (none leading, none trailing). Hoisted ESM and frontmatter are
    // removed from the tree before that wrap happens — they are not children of the emitted
    // fragment — so they do not count here either.
    let mut emitted = 0usize;
    for child in &root.children {
        match child {
            Node::Yaml(y) => frontmatter = parse_frontmatter_yaml(&y.value),
            Node::MdxjsEsm(esm) => {
                hoisted.push_str(&esm.value);
                hoisted.push('\n');
            }
            other => {
                // A node that renders NOTHING where it is written (a GFM footnote
                // definition — it reappears in the trailing section) is dropped from the
                // hast entirely, so it takes no separator either. Emitting one anyway put
                // a stray `{"\n"}` in front of every footnote section.
                let mut rendered = String::new();
                emitter.node(other, &mut rendered)?;
                if rendered.is_empty() {
                    continue;
                }
                if emitted > 0 {
                    body.push_str(EOL);
                }
                body.push_str(&rendered);
                emitted += 1;
            }
        }
    }
    // GFM footnotes: `mdast-util-to-hast` appends the collected definitions as a trailing
    // `<section class="footnotes">` AFTER the document body, in first-reference order — one
    // more root child, so it takes a separator like any other.
    emitter.footer(&mut body, emitted > 0)?;

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

/// Whether a node is a text node made only of whitespace — the class mdx-js's unravel
/// drops when it lifts JSX out of a paragraph (`/^[\t\r\n ]+$/`, plus the empty string).
fn is_blank_text(node: &Node) -> bool {
    matches!(node, Node::Text(t) if t.value.chars().all(|c| matches!(c, '\t' | '\r' | '\n' | ' ')))
}

/// mdx-js's `remark-mark-and-unravel`, in Rust.
///
/// A single-line `<Button>Hello</Button>` is NOT flow JSX to micromark (flow requires the
/// line to hold nothing but tags), so markdown-rs — like micromark, which markdown-rs
/// ports — parses it as a *paragraph* containing an `MdxJsxTextElement`. Emitting that
/// tree literally yields `<p><Button>Hello</Button></p>`, which is not what any MDX
/// compiler produces: `@mdx-js/mdx` (v1 through v3 alike) runs an "unravel" pass that
/// replaces a paragraph whose children are only JSX elements / MDX expressions and
/// whitespace with those children, promoted to flow. Without this pass every MDX page
/// that drops a component on its own line renders an extra, invalid `<p>` wrapper.
///
/// Applies at every depth (a paragraph inside a list item or blockquote unravels too),
/// exactly like the `unist-util-visit` walk it mirrors.
fn unravel(children: &mut Vec<Node>) {
    let mut index = 0;
    while index < children.len() {
        if let Some(grandchildren) = children[index].children_mut() {
            unravel(grandchildren);
        }
        let unravels = match &children[index] {
            Node::Paragraph(p) => {
                let mut has_jsx = false;
                let mut only_jsx_and_blanks = true;
                for child in &p.children {
                    match child {
                        Node::MdxJsxTextElement(_) | Node::MdxTextExpression(_) => has_jsx = true,
                        child if is_blank_text(child) => {}
                        _ => {
                            only_jsx_and_blanks = false;
                            break;
                        }
                    }
                }
                has_jsx && only_jsx_and_blanks
            }
            _ => false,
        };
        if !unravels {
            index += 1;
            continue;
        }
        let Node::Paragraph(paragraph) = children.remove(index) else {
            unreachable!()
        };
        let lifted: Vec<Node> = paragraph
            .children
            .into_iter()
            .filter(|c| !is_blank_text(c))
            .collect();
        let lifted_count = lifted.len();
        for (offset, node) in lifted.into_iter().enumerate() {
            children.insert(index + offset, node);
        }
        index += lifted_count;
    }
}

/// The mdast -> JSX emitter and its per-file state.
///
/// Beyond "which file" (for errors) and "route intrinsics through `_components`?" it holds
/// the GFM footnote bookkeeping that `mdast-util-to-hast` keeps on its `state`: the
/// definitions found anywhere in the tree, the order in which they were first *referenced*
/// (which is the order the trailing section lists them in, NOT document order), and how
/// many times each was referenced (which decides how many back-references it gets).
struct Emitter<'a> {
    path: &'a Path,
    use_map: bool,
    /// Every `FootnoteDefinition` in the document, keyed by UPPERCASED identifier — the
    /// same normalization `mdast-util-to-hast` uses, and first-one-wins on a duplicate.
    footnote_by_id: HashMap<String, &'a Node>,
    /// Referenced footnote identifiers, in first-reference order.
    footnote_order: Vec<String>,
    /// Identifier -> number of references emitted so far.
    footnote_counts: HashMap<String, usize>,
}

/// Collect every `FootnoteDefinition` in the tree (they are root-level in GFM, but a
/// definition nested in a blockquote is still collected, exactly as the reference
/// implementation's `unist-util-visit` pass does). First definition of an identifier wins.
fn collect_footnote_definitions<'a>(nodes: &'a [Node], out: &mut HashMap<String, &'a Node>) {
    for node in nodes {
        if let Node::FootnoteDefinition(def) = node {
            out.entry(def.identifier.to_uppercase()).or_insert(node);
        }
        if let Some(children) = node.children() {
            collect_footnote_definitions(children, out);
        }
    }
}

impl<'a> Emitter<'a> {
    fn new(path: &'a Path, use_map: bool, root_children: &'a [Node]) -> Emitter<'a> {
        let mut footnote_by_id = HashMap::new();
        collect_footnote_definitions(root_children, &mut footnote_by_id);
        Emitter {
            path,
            use_map,
            footnote_by_id,
            footnote_order: Vec::new(),
            footnote_counts: HashMap::new(),
        }
    }

    /// The JSX tag for an intrinsic element: the resolved-map member `_components.<tag>`
    /// when an `mdx-components.tsx` override is in play, otherwise the plain intrinsic.
    fn tag(&self, tag: &str) -> String {
        if self.use_map {
            format!("_components.{tag}")
        } else {
            tag.to_string()
        }
    }

    /// Emit one mdast node as JSX into `out`. Unhandled nodes hard-error (naming node+file).
    fn node(&mut self, node: &Node, out: &mut String) -> Result<(), String> {
        match node {
            Node::Heading(h) => self.wrap(&format!("h{}", h.depth), &h.children, out)?,
            Node::Paragraph(p) => self.wrap("p", &p.children, out)?,
            Node::Emphasis(e) => self.wrap("em", &e.children, out)?,
            Node::Strong(s) => self.wrap("strong", &s.children, out)?,
            // A blockquote is wrapped LOOSE (`wrap(children, true)`): a `{"\n"}` before the
            // first child, between children, and after the last.
            Node::Blockquote(b) => self.wrap_loose("blockquote", &b.children, out)?,
            // A bare list item outside a list cannot come out of the parser; judge its
            // looseness by itself, as `mdast-util-to-hast`'s `listItemLoose` does.
            Node::ListItem(li) => self.list_item(li, li.spread, out)?,
            Node::List(l) => {
                let loose = list_is_loose(l);
                let tag = self.tag(if l.ordered { "ol" } else { "ul" });
                // A list holding any GFM task item is marked, exactly as
                // `mdast-util-to-hast` marks it, so the usual `list-style: none` CSS for
                // task lists has something to hook onto.
                let task_list = l
                    .children
                    .iter()
                    .any(|child| matches!(child, Node::ListItem(li) if li.checked.is_some()));
                if task_list {
                    out.push_str(&format!("<{tag} className=\"contains-task-list\">"));
                } else {
                    out.push_str(&format!("<{tag}>"));
                }
                // A list is wrapped LOOSE, like a blockquote: `<ul>{"\n"}<li/>{"\n"}<li/>{"\n"}</ul>`.
                out.push_str(EOL);
                for (index, child) in l.children.iter().enumerate() {
                    if index > 0 {
                        out.push_str(EOL);
                    }
                    match child {
                        Node::ListItem(li) => self.list_item(li, loose, out)?,
                        other => self.node(other, out)?,
                    }
                }
                if !l.children.is_empty() {
                    out.push_str(EOL);
                }
                out.push_str(&format!("</{tag}>"));
            }
            // `href`/`src` are percent-normalized, as `mdast-util-to-hast` normalizes
            // them: a GFM autolink literal such as `www.a👍b.com` must reach the DOM as
            // `http://www.a%F0%9F%91%8Db.com`, not as raw UTF-8.
            Node::Link(l) => {
                let tag = self.tag("a");
                out.push_str(&format!(
                    "<{tag} href={}>",
                    js_string(&normalize_uri(&l.url))
                ));
                self.children(&l.children, out)?;
                out.push_str(&format!("</{tag}>"));
            }
            Node::Image(i) => {
                out.push_str(&format!(
                    "<{} src={} alt={} />",
                    self.tag("img"),
                    js_string(&normalize_uri(&i.url)),
                    js_string(&i.alt)
                ));
            }
            Node::Text(t) => out.push_str(&js_expr_string(&t.value)),
            Node::InlineCode(c) => {
                let tag = self.tag("code");
                out.push_str(&format!("<{tag}>{}</{tag}>", js_expr_string(&c.value)));
            }
            Node::Code(c) => {
                let lang = c.lang.as_deref().unwrap_or("");
                let class = if lang.is_empty() {
                    String::new()
                } else {
                    format!(" className={}", js_string(&format!("language-{lang}")))
                };
                let pre = self.tag("pre");
                let code = self.tag("code");
                // mdast strips the fence's closing newline from `value`; every markdown
                // pipeline puts it back (`mdast-util-to-hast`'s `code` handler appends
                // `\n` to a NON-empty value), and inside a `<pre>` that newline is
                // significant. An empty fence stays genuinely empty.
                if c.value.is_empty() {
                    out.push_str(&format!("<{pre}><{code}{class} /></{pre}>"));
                } else {
                    out.push_str(&format!(
                        "<{pre}><{code}{class}>{}</{code}></{pre}>",
                        js_expr_string(&format!("{}\n", c.value))
                    ));
                }
            }
            Node::ThematicBreak(_) => out.push_str(&format!("<{} />", self.tag("hr"))),
            Node::Break(_) => out.push_str(&format!("<{} />", self.tag("br"))),
            // --- GFM (only reachable when `Gfm::On` let these parse) ---
            Node::Delete(d) => self.wrap("del", &d.children, out)?,
            Node::Table(t) => self.table(t, out)?,
            Node::FootnoteReference(r) => self.footnote_reference(&r.identifier, out),
            // Definitions render nothing where they are written; they are collected and
            // emitted as the trailing footnotes section by `footer`.
            Node::FootnoteDefinition(_) => {}
            Node::MdxFlowExpression(e) => out.push_str(&format!("{{{}}}", e.value)),
            Node::MdxTextExpression(e) => out.push_str(&format!("{{{}}}", e.value)),
            Node::MdxJsxFlowElement(el) => {
                self.jsx_element(el.name.as_deref(), &el.attributes, &el.children, out)?
            }
            Node::MdxJsxTextElement(el) => {
                self.jsx_element(el.name.as_deref(), &el.attributes, &el.children, out)?
            }
            // MDX ESM only appears at the top level (hoisted in `compile`); a nested one is
            // malformed. Everything else is an explicitly unsupported node.
            other => {
                return Err(format!(
                    "MDX {}: unsupported node `{}` (diffpack's MDX emitter handles headings, paragraphs, \
                     emphasis/strong, links, images, lists, blockquotes, inline/fenced code, thematic breaks, \
                     MDX expressions, MDX JSX elements, and — when the app configures `remark-gfm` — tables, \
                     strikethrough, task lists, autolink literals and footnotes)",
                    self.path.display(),
                    node_kind(other),
                ));
            }
        }
        Ok(())
    }

    /// `<tag>children</tag>`, where `tag` is an intrinsic name mapped through `_components`.
    fn wrap(&mut self, tag: &str, children: &[Node], out: &mut String) -> Result<(), String> {
        let tag = self.tag(tag);
        out.push_str(&format!("<{tag}>"));
        self.children(children, out)?;
        out.push_str(&format!("</{tag}>"));
        Ok(())
    }

    fn children(&mut self, children: &[Node], out: &mut String) -> Result<(), String> {
        for child in children {
            self.node(child, out)?;
        }
        Ok(())
    }

    /// `<tag>` + `mdast-util-to-hast`'s LOOSE `wrap()` of the children: a [`EOL`] before the
    /// first, between each pair, and after the last (the trailing one only when there is at
    /// least one child, so an empty blockquote is `<blockquote>{"\n"}</blockquote>`).
    fn wrap_loose(&mut self, tag: &str, children: &[Node], out: &mut String) -> Result<(), String> {
        let tag = self.tag(tag);
        out.push_str(&format!("<{tag}>"));
        out.push_str(EOL);
        for (index, child) in children.iter().enumerate() {
            if index > 0 {
                out.push_str(EOL);
            }
            self.node(child, out)?;
        }
        if !children.is_empty() {
            out.push_str(EOL);
        }
        out.push_str(&format!("</{tag}>"));
        Ok(())
    }

    /// One `<li>`, mirroring `mdast-util-to-hast`'s `listItem` handler:
    ///
    /// * in a TIGHT list the item's paragraphs are unwrapped (`<li>text</li>`, not
    ///   `<li><p>text</p></li>`) — this is what every markdown-to-HTML pipeline does, and
    ///   without it a plain bullet list renders with block-level gaps the author never
    ///   wrote;
    /// * a GFM task item (`- [x] done`, i.e. `checked` is set) gains
    ///   `className="task-list-item"` and a disabled checkbox pushed to the front of its
    ///   first paragraph, followed by a space.
    fn list_item(
        &mut self,
        li: &markdown::mdast::ListItem,
        loose: bool,
        out: &mut String,
    ) -> Result<(), String> {
        let tag = self.tag("li");
        out.push('<');
        out.push_str(&tag);
        if li.checked.is_some() {
            out.push_str(" className=\"task-list-item\"");
        }
        out.push('>');

        // The checkbox belongs INSIDE the item's first paragraph (unwrapped again when the
        // list is tight), so `- [ ] a` renders `<li><input …/> a</li>`.
        // `checked` is a boolean attribute: present when true, ABSENT when false (the
        // reference emit drops a false boolean rather than writing `checked={false}`).
        let checkbox = li.checked.map(|checked| {
            format!(
                "<{} type=\"checkbox\"{} disabled />",
                self.tag("input"),
                if checked { " checked" } else { "" },
            )
        });
        // `mdast-util-to-hast` builds the item's `results` first: when the item is a task
        // whose first block is NOT a paragraph, a paragraph is SYNTHESIZED in front to hold
        // the checkbox — and that synthesized paragraph then takes part in the newline and
        // tight-list-unwrapping rules like any other.
        let first_is_paragraph = matches!(li.children.first(), Some(Node::Paragraph(_)));
        let synthesized = checkbox.is_some() && !first_is_paragraph;
        let paragraph = self.tag("p");
        let mut index = 0usize;

        if synthesized {
            let checkbox = checkbox.clone().unwrap_or_default();
            if eol_before_list_item_child(loose, index, true) {
                out.push_str(EOL);
            }
            if loose {
                out.push_str(&format!("<{paragraph}>{checkbox}</{paragraph}>"));
            } else {
                out.push_str(&checkbox);
            }
            index += 1;
        }

        for child in &li.children {
            let is_paragraph = matches!(child, Node::Paragraph(_));
            if eol_before_list_item_child(loose, index, is_paragraph) {
                out.push_str(EOL);
            }
            // The checkbox is merged into the first paragraph when there is one.
            let lead = if index == 0 && !synthesized {
                checkbox.as_deref()
            } else {
                None
            };
            match child {
                // A paragraph keeps its `<p>` only in a LOOSE list; in a tight one its
                // children are inlined straight into the `<li>`.
                Node::Paragraph(p) => {
                    if loose {
                        out.push_str(&format!("<{paragraph}>"));
                    }
                    if let Some(checkbox) = lead {
                        out.push_str(checkbox);
                        if !p.children.is_empty() {
                            out.push_str("{\" \"}");
                        }
                    }
                    self.children(&p.children, out)?;
                    if loose {
                        out.push_str(&format!("</{paragraph}>"));
                    }
                }
                other => self.node(other, out)?,
            }
            index += 1;
        }

        // The trailing newline, again from `mdast-util-to-hast`: present unless the last
        // child was a tight list item's unwrapped paragraph.
        let tail_is_paragraph = match li.children.last() {
            Some(node) => matches!(node, Node::Paragraph(_)),
            // The only child is the synthesized checkbox paragraph.
            None => synthesized,
        };
        if index > 0 && (loose || !tail_is_paragraph) {
            out.push_str(EOL);
        }
        out.push_str(&format!("</{tag}>"));
        Ok(())
    }

    /// A GFM table, laid out the way `mdast-util-to-hast` lays one out: the first row
    /// becomes a `<thead>` of `<th>`, the rest a `<tbody>` of `<td>`, and the delimiter
    /// row's alignment lands on every cell of that column. The column COUNT comes from the
    /// alignment row, so a short body row is padded with empty cells and a long one is
    /// truncated — same as GitHub.
    ///
    /// Alignment is emitted as `style={{ textAlign: ... }}`, not the `align` attribute
    /// mdast's hast carries: `hast-util-to-estree` (the JSX emitter `@mdx-js/mdx` runs)
    /// rewrites a table cell's `align` into exactly that style property, so this is what an
    /// author's own build puts in the DOM.
    ///
    /// No whitespace text nodes are emitted between the rows: `hast-util-to-estree` strips
    /// whitespace inside a table, where HTML would not render it anyway.
    fn table(&mut self, table: &markdown::mdast::Table, out: &mut String) -> Result<(), String> {
        use markdown::mdast::AlignKind;
        let align: Vec<Option<&str>> = table
            .align
            .iter()
            .map(|a| match a {
                AlignKind::Left => Some("left"),
                AlignKind::Right => Some("right"),
                AlignKind::Center => Some("center"),
                AlignKind::None => None,
            })
            .collect();

        let table_tag = self.tag("table");
        out.push_str(&format!("<{table_tag}>"));
        for (index, row) in table.children.iter().enumerate() {
            let Node::TableRow(row) = row else {
                return Err(format!(
                    "MDX {}: a GFM table row was parsed as `{}`",
                    self.path.display(),
                    node_kind(row),
                ));
            };
            let cell_tag = self.tag(if index == 0 { "th" } else { "td" });
            if index == 0 {
                out.push_str(&format!("<{}>", self.tag("thead")));
            } else if index == 1 {
                out.push_str(&format!("<{}>", self.tag("tbody")));
            }
            let tr = self.tag("tr");
            out.push_str(&format!("<{tr}>"));
            let columns = if align.is_empty() {
                row.children.len()
            } else {
                align.len()
            };
            for column in 0..columns {
                let attr = match align.get(column).copied().flatten() {
                    Some(value) => format!(" style={{{{textAlign: {}}}}}", js_string(value)),
                    None => String::new(),
                };
                let mut inner = String::new();
                if let Some(Node::TableCell(cell)) = row.children.get(column) {
                    self.children(&cell.children, &mut inner)?;
                }
                if inner.is_empty() {
                    // A missing or empty cell is a void element, not an empty pair.
                    out.push_str(&format!("<{cell_tag}{attr} />"));
                } else {
                    out.push_str(&format!("<{cell_tag}{attr}>{inner}</{cell_tag}>"));
                }
            }
            out.push_str(&format!("</{tr}>"));
            if index == 0 {
                out.push_str(&format!("</{}>", self.tag("thead")));
            }
        }
        if table.children.len() > 1 {
            out.push_str(&format!("</{}>", self.tag("tbody")));
        }
        out.push_str(&format!("</{table_tag}>"));
        Ok(())
    }

    /// `<sup><a href="#user-content-fn-x" id="user-content-fnref-x" …>N</a></sup>`, and
    /// record the reference so [`Emitter::footer`] can emit the definition and its
    /// back-references. A reference to a footnote that is never defined still renders (the
    /// footer simply has no entry for it) — that is GitHub's behaviour too.
    fn footnote_reference(&mut self, identifier: &str, out: &mut String) {
        let id = identifier.to_uppercase();
        let safe = normalize_uri(&id.to_lowercase());
        let counter = match self
            .footnote_order
            .iter()
            .position(|existing| existing == &id)
        {
            Some(index) => index + 1,
            None => {
                self.footnote_order.push(id.clone());
                self.footnote_order.len()
            }
        };
        let reuse = self.footnote_counts.entry(id).or_insert(0);
        *reuse += 1;
        let suffix = if *reuse > 1 {
            format!("-{reuse}")
        } else {
            String::new()
        };
        let sup = self.tag("sup");
        let anchor = self.tag("a");
        out.push_str(&format!(
            "<{sup}><{anchor} href=\"#{CLOBBER_PREFIX}fn-{safe}\" \
             id=\"{CLOBBER_PREFIX}fnref-{safe}{suffix}\" data-footnote-ref \
             aria-describedby=\"footnote-label\">{}</{anchor}></{sup}>",
            js_expr_string(&counter.to_string()),
        ));
    }

    /// The trailing `<section data-footnotes className="footnotes">` — nothing at all when
    /// no footnote was referenced. Definitions are listed in first-reference order and each
    /// gets one `↩` back-reference per reference to it. A definition referenced only from
    /// inside another footnote is picked up too (the loop re-reads `footnote_order`, which
    /// grows while it runs).
    fn footer(&mut self, out: &mut String, separate: bool) -> Result<(), String> {
        if self.footnote_order.is_empty() {
            return Ok(());
        }
        // The section is one more root-level child, so it is separated from the body the
        // same way two paragraphs are.
        if separate {
            out.push_str(EOL);
        }
        let mut items = String::new();
        let mut index = 0;
        let mut emitted_items = 0usize;
        while index < self.footnote_order.len() {
            let id = self.footnote_order[index].clone();
            index += 1;
            let Some(def) = self.footnote_by_id.get(&id).copied() else {
                continue;
            };
            let Node::FootnoteDefinition(def) = def else {
                unreachable!("footnote_by_id only holds FootnoteDefinition nodes")
            };
            let safe = normalize_uri(&id.to_lowercase());
            let li = self.tag("li");
            // The `<ol>` is a loose wrap and each definition is a LOOSE list item: a `{"\n"}`
            // between items, one before every child of an item, and one before its `</li>`.
            if emitted_items > 0 {
                items.push_str(EOL);
            }
            emitted_items += 1;
            items.push_str(&format!("<{li} id=\"{CLOBBER_PREFIX}fn-{safe}\">"));

            // Back-references are appended INSIDE the definition's last paragraph when it
            // ends in one, so the `↩` sits on the same line as the footnote text.
            let (blocks, tail) = match def.children.split_last() {
                Some((Node::Paragraph(p), head)) => (head, Some(&p.children)),
                _ => (def.children.as_slice(), None),
            };
            for block in blocks {
                items.push_str(EOL);
                self.node(block, &mut items)?;
            }
            let paragraph = self.tag("p");
            items.push_str(EOL);
            if let Some(children) = tail {
                items.push_str(&format!("<{paragraph}>"));
                // The separating space is MERGED into the trailing text run rather than
                // emitted as its own `{" "}` child: two adjacent React text children are
                // server-rendered with an `<!-- -->` marker between them, which the
                // author's own build does not produce.
                match children.split_last() {
                    Some((Node::Text(t), head)) => {
                        self.children(head, &mut items)?;
                        items.push_str(&js_expr_string(&format!("{} ", t.value)));
                    }
                    _ => {
                        self.children(children, &mut items)?;
                        items.push_str("{\" \"}");
                    }
                }
            }
            let count = self.footnote_counts.get(&id).copied().unwrap_or(0);
            let anchor = self.tag("a");
            let sup = self.tag("sup");
            for reference in 1..=count {
                if reference > 1 {
                    items.push_str("{\" \"}");
                }
                let suffix = if reference > 1 {
                    format!("-{reference}")
                } else {
                    String::new()
                };
                let label = if reference > 1 {
                    format!("Back to reference {}-{reference}", index)
                } else {
                    format!("Back to reference {index}")
                };
                items.push_str(&format!(
                    "<{anchor} href=\"#{CLOBBER_PREFIX}fnref-{safe}{suffix}\" \
                     data-footnote-backref=\"\" aria-label={} \
                     className=\"data-footnote-backref\">{}",
                    js_string(&label),
                    js_expr_string("\u{21a9}"),
                ));
                if reference > 1 {
                    items.push_str(&format!(
                        "<{sup}>{}</{sup}>",
                        js_expr_string(&reference.to_string())
                    ));
                }
                items.push_str(&format!("</{anchor}>"));
            }
            if tail.is_some() {
                items.push_str(&format!("</{paragraph}>"));
            }
            // The item's trailing newline (loose item).
            items.push_str(EOL);
            items.push_str(&format!("</{li}>"));
        }
        let section = self.tag("section");
        let heading = self.tag("h2");
        let ol = self.tag("ol");
        // `<section><h2/>{"\n"}<ol>{"\n"}…{"\n"}</ol>{"\n"}</section>` — the heading takes no
        // leading newline, everything after it does.
        out.push_str(&format!(
            "<{section} data-footnotes className=\"footnotes\">\
             <{heading} className=\"sr-only\" id=\"footnote-label\">{}</{heading}>\
             {EOL}<{ol}>{EOL}{items}{EOL}</{ol}>{EOL}</{section}>",
            js_expr_string("Footnotes"),
        ));
        Ok(())
    }

    /// Reconstruct an MDX JSX element `<Name attr="x" prop={expr} {...spread}>children</Name>`
    /// (a fragment when name is None).
    fn jsx_element(
        &mut self,
        name: Option<&str>,
        attributes: &[AttributeContent],
        children: &[Node],
        out: &mut String,
    ) -> Result<(), String> {
        let tag = name.unwrap_or("");
        out.push('<');
        out.push_str(tag);
        for attr in attributes {
            match attr {
                AttributeContent::Property(p) => match &p.value {
                    None => out.push_str(&format!(" {}", p.name)),
                    Some(AttributeValue::Literal(lit)) => {
                        out.push_str(&format!(" {}={}", p.name, js_string(lit)))
                    }
                    Some(AttributeValue::Expression(e)) => {
                        out.push_str(&format!(" {}={{{}}}", p.name, e.value))
                    }
                },
                AttributeContent::Expression(e) => out.push_str(&format!(" {{{}}}", e.value)),
            }
        }
        if children.is_empty() {
            out.push_str(" />");
            return Ok(());
        }
        out.push('>');
        self.children(children, out)?;
        out.push_str(&format!("</{tag}>"));
        Ok(())
    }
}

/// `mdast-util-to-hast`'s `listLoose`: a list is loose when it, or any of its items, is
/// "spread" (its blocks separated by blank lines). A tight list unwraps its items'
/// paragraphs; a loose one keeps them.
fn list_is_loose(list: &markdown::mdast::List) -> bool {
    list.spread
        || list
            .children
            .iter()
            .any(|child| matches!(child, Node::ListItem(li) if li.spread))
}

/// micromark's `normalizeUri`, used for footnote ids/hrefs: percent-encode everything a URL
/// cannot carry literally, leave existing percent-escapes alone.
fn normalize_uri(value: &str) -> String {
    fn keep(c: char) -> bool {
        // micromark's allow-list: /[!#$&-;=?-Z_a-z~]/
        matches!(c, '!' | '#' | '$' | '&'..=';' | '=' | '?'..='Z' | '_' | 'a'..='z' | '~')
    }
    let bytes: Vec<char> = value.chars().collect();
    let mut out = String::with_capacity(value.len());
    let mut index = 0;
    while index < bytes.len() {
        let c = bytes[index];
        if c == '%'
            && bytes
                .get(index + 1)
                .is_some_and(|c| c.is_ascii_alphanumeric())
            && bytes
                .get(index + 2)
                .is_some_and(|c| c.is_ascii_alphanumeric())
        {
            out.push(c);
            out.push(bytes[index + 1]);
            out.push(bytes[index + 2]);
            index += 3;
            continue;
        }
        if c.is_ascii() && keep(c) {
            out.push(c);
        } else {
            let mut buf = [0u8; 4];
            for byte in c.encode_utf8(&mut buf).as_bytes() {
                out.push_str(&format!("%{byte:02X}"));
            }
        }
        index += 1;
    }
    out
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
        && let Some(stem) = Path::new(last.as_str())
            .file_stem()
            .and_then(|s| s.to_str())
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
        Node::Table(_) => "Table",
        Node::TableRow(_) => "TableRow",
        Node::TableCell(_) => "TableCell",
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

    /// Compile with GFM on, the way an app that configures `remark-gfm` gets compiled.
    fn gfm(src: &str) -> String {
        compile_native(Path::new("page.mdx"), src, Gfm::On)
            .unwrap()
            .jsx
    }

    /// Compile with GFM off — the `@next/mdx` default, and what an app that did NOT
    /// configure `remark-gfm` must keep getting.
    fn commonmark(src: &str) -> String {
        compile_native(Path::new("page.mdx"), src, Gfm::Off)
            .unwrap()
            .jsx
    }

    #[test]
    fn detects_mdx_paths() {
        assert!(is_mdx_path(Path::new("a/page.mdx")));
        assert!(is_mdx_path(Path::new("a/readme.md")));
        assert!(!is_mdx_path(Path::new("a/page.tsx")));
    }

    #[test]
    fn frontmatter_becomes_metadata_not_body() {
        let out = compile(
            Path::new("p.mdx"),
            "---\ntitle: Hi\ndescription: D\n---\n\n# H\n",
        )
        .unwrap();
        assert_eq!(out.frontmatter.get("title").map(String::as_str), Some("Hi"));
        assert_eq!(
            out.frontmatter.get("description").map(String::as_str),
            Some("D")
        );
        assert!(
            !out.jsx.contains("title: Hi"),
            "frontmatter is not in the body: {}",
            out.jsx
        );
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
        // Content is a JS string child (so `<`/`{` need no JSX escaping) and byte-preserved,
        // down to the closing fence's newline (significant inside a `<pre>`).
        assert!(out.contains("{\"const a = 1 < 2;\\n\"}"), "{out}");
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
    fn scaffold_with_components(
        components_rel_dir: &str,
    ) -> (std::path::PathBuf, std::path::PathBuf) {
        let mut root = std::env::temp_dir();
        root.push(format!(
            "diffpack-mdx-{}-{}",
            std::process::id(),
            components_rel_dir.replace('/', "_")
        ));
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
        assert!(
            out.contains("export default function MDXContent() {"),
            "{out}"
        );
        assert!(!out.contains("_components"), "{out}");
        assert!(!out.contains("_provideComponents"), "{out}");
    }

    #[test]
    fn root_components_file_routes_intrinsics_through_map() {
        let (root, page) = scaffold_with_components(".");
        let out = compile(&page, "# Hi\n\nA [link](/x) and `code`.\n")
            .unwrap()
            .jsx;
        std::fs::remove_dir_all(&root).ok();
        // Imports the app override and resolves the map once.
        assert!(
            out.contains(
                "import { useMDXComponents as _provideComponents } from \"../../mdx-components\""
            ),
            "{out}"
        );
        assert!(out.contains("const _components = {"), "{out}");
        assert!(out.contains("h1: \"h1\""), "{out}");
        assert!(out.contains("..._provideComponents()"), "{out}");
        assert!(
            out.contains("export default function MDXContent(props)"),
            "{out}"
        );
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
        assert_eq!(
            relative_import_specifier(from, target),
            "../../../mdx-components"
        );

        let sibling = Path::new("/proj/app/blog/mdx-components.ts");
        assert_eq!(
            relative_import_specifier(Path::new("/proj/app/blog/page.mdx"), sibling),
            "./mdx-components"
        );
    }

    // --- FINDINGS #31: MDX wrapped a stand-alone component in a spurious <p> ---------

    #[test]
    fn a_component_on_its_own_line_is_not_wrapped_in_a_paragraph() {
        // `next-pages-mdx`'s own page. Both @mdx-js/loader v1 (what the app pins) and
        // @mdx-js/mdx v3 emit `<h1/><p/><Button/>` as siblings; diffpack emitted
        // `<p><Button>…</Button></p>` because micromark (and markdown-rs, its port)
        // only calls a JSX line "flow" when the line holds nothing but tags.
        let out = jsx("# MDX + Next.js\n\nLook, a button!\n\n<Button>Hello</Button>\n");
        assert!(out.contains("<Button>{\"Hello\"}</Button>"), "{out}");
        assert!(
            !out.contains("<p><Button>"),
            "a stand-alone component must not be wrapped in a paragraph: {out}"
        );
        // The real paragraph is untouched.
        assert!(out.contains("<p>{\"Look, a button!\"}</p>"), "{out}");
    }

    #[test]
    fn prose_around_an_inline_component_keeps_its_paragraph() {
        // Unravel is only for paragraphs made ENTIRELY of JSX/expressions + whitespace.
        let out = jsx("See <Button>Hi</Button> there\n");
        assert!(
            out.contains("<p>"),
            "inline JSX stays inside its paragraph: {out}"
        );
        assert!(out.contains("<Button>{\"Hi\"}</Button>"), "{out}");
    }

    #[test]
    fn several_components_on_one_line_all_unravel_and_lose_the_gap_text() {
        let out = jsx("<A>x</A> <B>y</B>\n");
        assert!(!out.contains("<p>"), "{out}");
        // Both are promoted to FLOW, so they are separated by the root wrap's `{"\n"}` —
        // which is also what makes them render with a space between them, as `@mdx-js/mdx`
        // renders them.
        assert!(
            out.contains("<A>{\"x\"}</A>{\"\\n\"}<B>{\"y\"}</B>"),
            "{out}"
        );
    }

    #[test]
    fn unravel_reaches_nested_blocks() {
        // The mdx-js pass is a whole-tree visit, not a root-level scan.
        let out = jsx("> <Note>hi</Note>\n");
        assert!(
            out.contains("<blockquote>{\"\\n\"}<Note>{\"hi\"}</Note>{\"\\n\"}</blockquote>"),
            "{out}"
        );
    }

    #[test]
    fn a_stand_alone_mdx_expression_unravels_too() {
        // `remark-mark-and-unravel` promotes mdxTextExpression the same way.
        let out = jsx("{greeting}\n");
        assert!(!out.contains("<p>"), "{out}");
        assert!(out.contains("{greeting}"), "{out}");
    }

    #[test]
    fn unsupported_node_is_a_clear_hard_error() {
        // A reference-style link definition is not in the supported subset — a clear
        // error naming the node, never a silent drop.
        let err = compile(Path::new("t.mdx"), "[ref]: https://example.com\n").unwrap_err();
        assert!(
            err.contains("unsupported node") && err.contains("Definition"),
            "{err}"
        );
    }

    // -----------------------------------------------------------------------------------
    // `createMDX({ options: { remarkPlugins, rehypePlugins } })`
    //
    // The defect: these options were read by NOTHING. An app configuring `remarkGfm` got
    // plain CommonMark — no GFM tables, no strikethrough — with no build warning and no
    // runtime error. The page just rendered differently from what the author wrote.
    // -----------------------------------------------------------------------------------

    fn eval_json(mdx: serde_json::Value) -> serde_json::Value {
        serde_json::json!({ "mdx": mdx })
    }

    #[test]
    fn createmdx_plugin_options_are_read_not_dropped() {
        let eval = eval_json(serde_json::json!({
            "configured": true,
            "remarkPlugins": [{ "name": "remarkGfm", "kind": "function", "hasOptions": false }],
            "rehypePlugins": [{ "name": "rehype-slug", "kind": "specifier", "hasOptions": true }],
            "recmaPlugins": [],
            "providerImportSource": null,
            "extension": null,
            "mdxRs": false,
            "otherOptions": ["format"],
        }));
        let config = MdxConfig::from_eval(Some(&eval));
        assert!(config.configured);
        assert_eq!(config.remark_plugins.len(), 1);
        assert_eq!(config.remark_plugins[0].name, "remarkGfm");
        assert_eq!(config.rehype_plugins[0].kind, "specifier");
        assert!(config.rehype_plugins[0].has_options);

        // Every configured plugin diffpack cannot run itself is named, so a build error /
        // log line can quote them. `remarkGfm` is NOT among them — it is implemented
        // natively — but it still shows up in the summary so the log records that diffpack
        // saw it and what it did about it.
        let unhonored = config.unhonored_options().join(" | ");
        assert!(
            !unhonored.contains("remarkGfm"),
            "gfm is native now: {unhonored}"
        );
        assert!(
            unhonored.contains("rehype-slug (with options)"),
            "{unhonored}"
        );
        assert!(unhonored.contains("options.format"), "{unhonored}");
        assert!(config.wants_gfm());
        assert!(
            config.summary().contains("remark-gfm (native GFM)"),
            "{}",
            config.summary()
        );
    }

    #[test]
    fn remark_gfm_alone_is_honoured_natively_rather_than_deferred() {
        // The whole point of the GFM work: an app whose only plugin is `remark-gfm` gets
        // real tables/strikethrough WITHOUT a node process per file, and without needing
        // `@mdx-js/mdx` installed at all.
        for name in ["remark-gfm", "remarkGfm", "gfm"] {
            let config = MdxConfig::from_eval(Some(&eval_json(serde_json::json!({
                "configured": true,
                "remarkPlugins": [{ "name": name, "kind": "specifier", "hasOptions": false }],
            }))));
            assert!(config.wants_gfm(), "{name} is remark-gfm");
            assert!(
                config.unhonored_options().is_empty(),
                "{name} must not be deferred: {:?}",
                config.unhonored_options()
            );
        }
    }

    #[test]
    fn remark_gfm_with_options_still_defers_to_the_apps_own_pipeline() {
        // `[remarkGfm, { singleTilde: false }]` changes what parses, and only the plugin's
        // IDENTITY survives the config eval — the options object does not. Guessing would
        // be a silent divergence, so this shape stays on the app's own pipeline.
        let config = MdxConfig::from_eval(Some(&eval_json(serde_json::json!({
            "configured": true,
            "remarkPlugins": [{ "name": "remark-gfm", "kind": "specifier", "hasOptions": true }],
        }))));
        assert!(!config.wants_gfm());
        assert!(
            config
                .unhonored_options()
                .join(" ")
                .contains("remark-gfm (with options)"),
            "{:?}",
            config.unhonored_options()
        );
    }

    #[test]
    fn remark_gfm_beside_another_plugin_still_defers_the_whole_file() {
        // Native GFM cannot be composed with an arbitrary unified plugin, so the file goes
        // to the app's pipeline — which runs BOTH, gfm included.
        let config = MdxConfig::from_eval(Some(&eval_json(serde_json::json!({
            "configured": true,
            "remarkPlugins": [
                { "name": "remark-gfm", "kind": "specifier", "hasOptions": false },
                { "name": "remark-toc", "kind": "specifier", "hasOptions": false },
            ],
        }))));
        let unhonored = config.unhonored_options().join(" | ");
        assert!(unhonored.contains("remark-toc"), "{unhonored}");
        assert!(!unhonored.contains("remark-gfm"), "{unhonored}");
        assert!(
            !config.unhonored_options().is_empty(),
            "the file is still deferred"
        );
    }

    #[test]
    fn createmdx_without_plugins_keeps_the_native_compiler() {
        // The shape both pinned MDX fixtures use: `createMDX()` (app router, plus
        // `experimental.mdxRs`) and `createMDX({ extension: /\.mdx?$/ })` (pages router).
        // Neither asks for anything the native emitter cannot do, so neither may be pushed
        // onto the node pipeline (or hard-errored).
        for mdx in [
            serde_json::json!({ "configured": true, "mdxRs": true }),
            serde_json::json!({ "configured": true, "extension": "/\\.mdx?$/" }),
        ] {
            let config = MdxConfig::from_eval(Some(&eval_json(mdx)));
            assert!(config.configured);
            assert!(
                config.unhonored_options().is_empty(),
                "native compiler is faithful here: {:?}",
                config.unhonored_options()
            );
        }
        // No `mdx` block at all (no next.config, an app that does not use @next/mdx).
        assert!(!MdxConfig::from_eval(None).configured);
        assert!(MdxConfig::from_eval(None).unhonored_options().is_empty());
    }

    /// A minimal Next project at `root` whose next.config passes a plugin diffpack has no
    /// native implementation of (`remarkToc`) to `createMDX`, plus one `.mdx` page. Returns
    /// the page path. Deliberately NOT `remark-gfm`: that one is now implemented natively,
    /// so it would never reach the app's pipeline.
    fn scratch_mdx_project(root: &Path) -> PathBuf {
        std::fs::write(root.join("package.json"), r#"{"name":"scratch"}"#).unwrap();
        std::fs::write(
            root.join("next.config.js"),
            "function remarkToc() {}\n\
             const withMDX = require('@next/mdx')({ options: { remarkPlugins: [remarkToc] } });\n\
             module.exports = withMDX({ pageExtensions: ['tsx', 'mdx'] });\n",
        )
        .unwrap();
        let page = root.join("page.mdx");
        std::fs::write(&page, "# Hi\n").unwrap();
        page
    }

    /// A stand-in `@mdx-js/mdx` that reports back which options it was handed, so the test
    /// can prove the app's configured plugins really reach the compiler rather than
    /// asserting on a hard-coded pipeline.
    fn install_stub_mdx_compiler(root: &Path) {
        let dir = root.join("node_modules/@mdx-js/mdx");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("package.json"),
            r#"{"name":"@mdx-js/mdx","version":"3.0.0","type":"module","exports":"./index.js"}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("index.js"),
            "export async function compile(vfile, options) {\n\
               const names = (options.remarkPlugins || [])\n\
                 .map((p) => (typeof p === 'function' ? p.name : String(p)))\n\
                 .join(',');\n\
               return `export default function MDXContent() { return <p data-remark=\"${names}\" \
             data-jsx=\"${options.jsx}\" data-provider=\"${options.providerImportSource || ''}\">\
             ${vfile.value.trim()}</p>; }`;\n\
             }\n",
        )
        .unwrap();
    }

    #[test]
    fn configured_remark_plugins_reach_the_apps_own_mdx_compiler() {
        if Command::new("node").arg("--version").output().is_err() {
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let page = scratch_mdx_project(root);
        install_stub_mdx_compiler(root);
        std::fs::write(
            root.join("mdx-components.tsx"),
            "export function useMDXComponents() { return {}; }\n",
        )
        .unwrap();

        let compiled = compile(&page, "# Hi\n").unwrap();
        // The configured plugin was passed through, by identity.
        assert!(
            compiled.jsx.contains("data-remark=\"remarkToc\""),
            "the app's remarkPlugins must reach its compiler: {}",
            compiled.jsx
        );
        // diffpack keeps the JSX emit (its own oxc pipeline compiles it downstream)...
        assert!(
            compiled.jsx.contains("data-jsx=\"true\""),
            "{}",
            compiled.jsx
        );
        // ...and the app's `mdx-components.tsx` still supplies the component overrides,
        // standing in for @next/mdx's `next-mdx-import-source-file` alias.
        assert!(
            compiled.jsx.contains("mdx-components.tsx\""),
            "providerImportSource must point at the app's mdx-components: {}",
            compiled.jsx
        );
    }

    #[test]
    fn an_esm_next_config_is_captured_too() {
        if Command::new("node").arg("--version").output().is_err() {
            return;
        }
        // `import createMDX from "@next/mdx"` in a `next.config.mjs` never goes through
        // `Module._load`, so the CJS-only interception saw nothing and the options vanished.
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        std::fs::write(
            root.join("package.json"),
            r#"{"name":"scratch","type":"module"}"#,
        )
        .unwrap();
        std::fs::write(
            root.join("next.config.mjs"),
            "import createMDX from '@next/mdx';\n\
             function remarkToc() {}\n\
             const withMDX = createMDX({ options: { remarkPlugins: [remarkToc] } });\n\
             export default withMDX({ pageExtensions: ['tsx', 'mdx'] });\n",
        )
        .unwrap();
        let page = root.join("page.mdx");
        std::fs::write(&page, "# Hi\n").unwrap();
        install_stub_mdx_compiler(root);

        let compiled = compile(&page, "# Hi\n").unwrap();
        assert!(
            compiled.jsx.contains("data-remark=\"remarkToc\""),
            "an ESM next.config's createMDX options must be captured: {}",
            compiled.jsx
        );
    }

    #[test]
    fn configured_plugins_without_an_app_pipeline_are_a_hard_error() {
        if Command::new("node").arg("--version").output().is_err() {
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let page = scratch_mdx_project(root);
        // No `@mdx-js/mdx` installed: diffpack cannot run `remarkToc`, and compiling
        // without it would silently render something else. It must refuse, naming both.
        let error = compile(&page, "# Hi\n").unwrap_err();
        assert!(error.contains("remarkToc"), "names the plugin: {error}");
        assert!(error.contains("page.mdx"), "names the file: {error}");
        assert!(
            error.contains("@mdx-js/mdx"),
            "names what is missing: {error}"
        );
    }

    // -----------------------------------------------------------------------------------
    // GitHub-Flavoured Markdown (`remark-gfm`)
    //
    // The defect: diffpack's MDX was CommonMark only. `~~struck~~` rendered as the literal
    // characters, a pipe table rendered as a paragraph of pipes, `- [ ] task` rendered as a
    // plain bullet, and a bare `www.example.com` stayed prose — half of what people mean by
    // "markdown" was silently dropped.
    //
    // Every expectation below was taken from the app's OWN toolchain: `@mdx-js/mdx` 3 with
    // `remark-gfm`, compiled with `{jsx: true, outputFormat: "program"}` — the exact call
    // `mdx_runner.mjs` makes. The one systematic difference is the `{"\n"}` whitespace text
    // nodes that `mdast-util-to-hast` puts between block children, which this emitter has
    // never emitted for ANY construct (they are inter-element whitespace, not content).
    // -----------------------------------------------------------------------------------

    #[test]
    fn gfm_tables_render_with_alignment() {
        let out = gfm("| a | b | c |\n| :- | --: | :-: |\n| 1 | 2 | 3 |\n");
        assert!(
            out.contains(
                "<table><thead><tr>\
                 <th style={{textAlign: \"left\"}}>{\"a\"}</th>\
                 <th style={{textAlign: \"right\"}}>{\"b\"}</th>\
                 <th style={{textAlign: \"center\"}}>{\"c\"}</th>\
                 </tr></thead><tbody><tr>\
                 <td style={{textAlign: \"left\"}}>{\"1\"}</td>\
                 <td style={{textAlign: \"right\"}}>{\"2\"}</td>\
                 <td style={{textAlign: \"center\"}}>{\"3\"}</td>\
                 </tr></tbody></table>"
            ),
            "{out}"
        );
    }

    #[test]
    fn gfm_table_without_alignment_has_no_style_and_a_header_only_table_has_no_tbody() {
        let out = gfm("| a |\n| - |\n| 1 |\n");
        assert!(!out.contains("style="), "no alignment was requested: {out}");
        assert!(
            out.contains("<thead><tr><th>{\"a\"}</th></tr></thead>"),
            "{out}"
        );
        let head_only = gfm("| a |\n| - |\n");
        assert!(
            !head_only.contains("tbody"),
            "no body rows, no tbody: {head_only}"
        );
    }

    #[test]
    fn gfm_table_rows_are_padded_and_truncated_to_the_header_width() {
        // GitHub sizes every row by the delimiter row: a short row gains empty cells, a
        // long one loses the extras.
        let short = gfm("| a | b |\n| - | - |\n| 1 |\n");
        assert!(short.contains("<td>{\"1\"}</td><td /></tr>"), "{short}");
        let long = gfm("| a | b |\n| - | - |\n| 1 | 2 | 3 |\n");
        assert!(
            !long.contains("{\"3\"}"),
            "the third cell has no column: {long}"
        );
    }

    #[test]
    fn gfm_table_cells_keep_their_inline_markup() {
        let out = gfm("| a |\n| - |\n| **x** and [l](/u) |\n");
        assert!(out.contains("<strong>{\"x\"}</strong>"), "{out}");
        assert!(out.contains("<a href=\"/u\">{\"l\"}</a>"), "{out}");
    }

    #[test]
    fn gfm_strikethrough_becomes_del() {
        let out = gfm("a ~~struck~~ b\n");
        assert!(
            out.contains("{\"a \"}<del>{\"struck\"}</del>{\" b\"}"),
            "{out}"
        );
        // Nested inside other inline markup too.
        let nested = gfm("**bold ~~and struck~~**\n");
        assert!(
            nested.contains("<strong>{\"bold \"}<del>{\"and struck\"}</del></strong>"),
            "{nested}"
        );
    }

    #[test]
    fn gfm_task_lists_get_a_disabled_checkbox_and_the_reference_class_names() {
        let out = gfm("- [ ] todo\n- [x] done\n");
        assert!(
            out.contains(
                "<ul className=\"contains-task-list\">{\"\\n\"}\
                 <li className=\"task-list-item\"><input type=\"checkbox\" disabled />{\" \"}{\"todo\"}</li>{\"\\n\"}\
                 <li className=\"task-list-item\"><input type=\"checkbox\" checked disabled />{\" \"}{\"done\"}</li>{\"\\n\"}\
                 </ul>"
            ),
            "{out}"
        );
        // A list with no task item is not marked.
        assert!(!gfm("- a\n").contains("contains-task-list"));
    }

    #[test]
    fn gfm_task_item_with_several_blocks_keeps_its_paragraphs() {
        // The checkbox belongs to the item's FIRST paragraph; a loose item keeps its
        // paragraph wrappers.
        let out = gfm("- [x] one\n\n  two\n");
        assert!(
            out.contains(
                "<li className=\"task-list-item\">{\"\\n\"}\
                 <p><input type=\"checkbox\" checked disabled />{\" \"}{\"one\"}</p>{\"\\n\"}\
                 <p>{\"two\"}</p>{\"\\n\"}</li>"
            ),
            "{out}"
        );
    }

    #[test]
    fn gfm_autolink_literals_become_links_with_normalized_urls() {
        let out = gfm("Visit www.example.com and https://a.b now.\n");
        assert!(
            out.contains("<a href=\"http://www.example.com\">{\"www.example.com\"}</a>"),
            "{out}"
        );
        assert!(
            out.contains("<a href=\"https://a.b\">{\"https://a.b\"}</a>"),
            "{out}"
        );
        // Email autolinks too.
        assert!(
            gfm("Mail a@b.com.\n").contains("href=\"mailto:a@b.com\""),
            "{}",
            gfm("Mail a@b.com.\n")
        );
        // A non-ASCII authority reaches the DOM percent-encoded, as `normalizeUri` encodes
        // it — not as raw UTF-8.
        let emoji = gfm("www.a\u{1f44d}b.com\n");
        assert!(
            emoji.contains("href=\"http://www.a%F0%9F%91%8Db.com\""),
            "{emoji}"
        );
    }

    #[test]
    fn gfm_footnotes_render_a_labelled_section_with_back_references() {
        let out = gfm("Note[^1] and again[^1].\n\n[^1]: The *note*.\n");
        // The reference: a numbered sup-link that points at the definition.
        assert!(
            out.contains(
                "<sup><a href=\"#user-content-fn-1\" id=\"user-content-fnref-1\" \
                 data-footnote-ref aria-describedby=\"footnote-label\">{\"1\"}</a></sup>"
            ),
            "{out}"
        );
        // The second reference to the SAME footnote reuses the number and takes a -2 id.
        assert!(out.contains("id=\"user-content-fnref-1-2\""), "{out}");
        // The section, its screen-reader label, and one back-reference per reference.
        assert!(
            out.contains(
                "<section data-footnotes className=\"footnotes\">\
                 <h2 className=\"sr-only\" id=\"footnote-label\">{\"Footnotes\"}</h2>\
                 {\"\\n\"}<ol>{\"\\n\"}<li id=\"user-content-fn-1\">{\"\\n\"}"
            ),
            "{out}"
        );
        // The separating space is merged into the trailing text run, not emitted as its own
        // child: two adjacent React text children are server-rendered with an `<!-- -->`
        // marker between them, which the author's own build does not produce.
        assert!(
            out.contains("<em>{\"note\"}</em>{\". \"}<a href=\"#user-content-fnref-1\""),
            "{out}"
        );
        assert!(
            out.contains(
                "aria-label=\"Back to reference 1\" className=\"data-footnote-backref\">{\"\u{21a9}\"}</a>"
            ),
            "{out}"
        );
        assert!(
            out.contains("href=\"#user-content-fnref-1-2\"")
                && out.contains("aria-label=\"Back to reference 1-2\"")
                && out.contains("{\"\u{21a9}\"}<sup>{\"2\"}</sup>"),
            "the second back-reference is numbered: {out}"
        );
    }

    #[test]
    fn gfm_footnotes_are_ordered_by_first_reference_and_unreferenced_ones_are_dropped() {
        let out = gfm("One[^b] two[^a].\n\n[^a]: Ay.\n\n[^b]: Bee.\n\n[^c]: Unused.\n");
        let bee = out
            .find("Bee")
            .expect("first-referenced definition is emitted first");
        let ay = out
            .find("Ay")
            .expect("second-referenced definition follows");
        assert!(bee < ay, "reference order, not document order: {out}");
        assert!(
            !out.contains("Unused"),
            "an unreferenced definition renders nothing: {out}"
        );
    }

    #[test]
    fn a_footnote_call_without_a_definition_stays_literal_text() {
        // GFM only forms a footnote call when a matching definition exists, so this must
        // NOT grow a dangling `<sup>` link or an empty footnotes section.
        let out = gfm("Missing[^gone] here.\n");
        assert!(out.contains("{\"Missing[^gone] here.\"}"), "{out}");
        assert!(!out.contains("footnotes"), "{out}");
    }

    #[test]
    fn gfm_footnote_definition_ending_in_a_block_takes_the_backref_outside_a_paragraph() {
        let out = gfm("Ref[^y].\n\n[^y]:\n    ```js\n    code\n    ```\n");
        assert!(
            out.contains("</pre>{\"\\n\"}<a href=\"#user-content-fnref-y\""),
            "no synthetic paragraph, and no separating space \u{2014} the backref is its own \
             loose-item child, so it takes a newline and not a space child: {out}"
        );
    }

    // --- `mdast-util-to-hast`'s `{"\n"}` block separators -------------------------------
    //
    // Found by the first-party e2e fixture `next-mdx-features`: two flow-level components on
    // consecutive lines rendered as `…routeclicked 0 times` under diffpack and
    // `…route clicked 0 times` under `next build`. Every expectation below was taken from
    // the REAL `@mdx-js/mdx@3` + `remark-gfm` at `{jsx: true, outputFormat: "program"}` —
    // the exact call `mdx_runner.mjs` makes — not from memory.

    #[test]
    fn root_children_are_separated_by_a_newline_child() {
        let out = commonmark("First paragraph.\n\nSecond paragraph.\n");
        assert!(
            out.contains("<p>{\"First paragraph.\"}</p>{\"\\n\"}<p>{\"Second paragraph.\"}</p>"),
            "{out}"
        );
        // ...but a single root child gets no separator at all, leading or trailing.
        let single = commonmark("Only one.\n");
        assert!(single.contains("(<><p>{\"Only one.\"}</p></>)"), "{single}");
    }

    #[test]
    fn two_flow_components_are_separated_the_way_mdx_separates_them() {
        // The exact shape the e2e fixture caught: `<Badge/>` then `<Counter/>`, each on its
        // own line. Both are inline elements, so the separator is the difference between
        // "badge inside an MDX route clicked 0 times" and "…routeclicked 0 times".
        let out = jsx("<Badge>a</Badge>\n\n<Counter />\n");
        assert!(
            out.contains("<Badge>{\"a\"}</Badge>{\"\\n\"}<Counter />"),
            "{out}"
        );
    }

    #[test]
    fn a_footnote_definition_takes_no_separator_where_it_is_written() {
        // A definition renders NOTHING in place (it reappears in the trailing section), so
        // it is dropped from the tree and must not leave a `{"\n"}` behind: emitting one
        // put a stray extra newline in front of every footnotes section.
        let out = gfm("Ref[^a].\n\n[^a]: The note.\n");
        assert!(
            !out.contains("{\"\\n\"}{\"\\n\"}"),
            "no doubled separator: {out}"
        );
        assert!(
            out.contains("</p>{\"\\n\"}<section data-footnotes"),
            "exactly one separator before the section: {out}"
        );
    }

    #[test]
    fn blockquotes_are_wrapped_loose() {
        let out = commonmark("> quoted\n>\n> more\n");
        assert!(
            out.contains(
                "<blockquote>{\"\\n\"}<p>{\"quoted\"}</p>{\"\\n\"}<p>{\"more\"}</p>{\"\\n\"}</blockquote>"
            ),
            "{out}"
        );
        // An empty blockquote keeps the LEADING newline and gains no trailing one, exactly
        // as `wrap([], true)` does.
        let empty = commonmark("> \n\ntext\n");
        assert!(
            empty.contains("<blockquote>{\"\\n\"}</blockquote>"),
            "{empty}"
        );
    }

    #[test]
    fn a_table_carries_no_separators_at_all() {
        // The negative that keeps the rule honest: `mdast-util-to-hast` wraps lists and
        // blockquotes, and does NOT wrap table rows or cells.
        let out = gfm("| a | b |\n| - | - |\n| 1 | 2 |\n");
        assert!(
            !out.contains("{\"\\n\"}"),
            "a table has no newline children: {out}"
        );
    }

    #[test]
    fn commonmark_leaves_every_gfm_construct_alone() {
        // The other half of the contract: an app that did NOT configure `remark-gfm` must
        // keep getting exactly what `@next/mdx` gives it. Enabling GFM unconditionally
        // would diverge from the app's own toolchain just as badly as dropping it.
        let table = commonmark("| a | b |\n| - | - |\n| 1 | 2 |\n");
        assert!(!table.contains("<table>"), "{table}");
        let struck = commonmark("~~struck~~\n");
        assert!(!struck.contains("<del>"), "{struck}");
        assert!(struck.contains("~~struck~~"), "{struck}");
        let task = commonmark("- [ ] task\n");
        assert!(!task.contains("checkbox"), "{task}");
        assert!(task.contains("[ ] task"), "{task}");
        let autolink = commonmark("Visit www.example.com now.\n");
        assert!(!autolink.contains("<a "), "{autolink}");
        let footnote = commonmark("Note[^1] here.\n");
        assert!(!footnote.contains("footnotes"), "{footnote}");
        assert!(footnote.contains("Note[^1] here."), "{footnote}");
    }

    #[test]
    fn tight_lists_unwrap_their_paragraphs_and_loose_lists_keep_them() {
        // `mdast-util-to-hast`'s listItem rule. Without it every bullet list renders with
        // block-level gaps the author never wrote — and a task item's checkbox lands inside
        // a stray `<p>`.
        let tight = commonmark("- a\n- b\n");
        assert!(
            tight.contains("<ul>{\"\\n\"}<li>{\"a\"}</li>{\"\\n\"}<li>{\"b\"}</li>{\"\\n\"}</ul>"),
            "{tight}"
        );
        let loose = commonmark("- a\n\n- b\n");
        assert!(
            loose.contains(
                "<ul>{\"\\n\"}<li>{\"\\n\"}<p>{\"a\"}</p>{\"\\n\"}</li>{\"\\n\"}\
                 <li>{\"\\n\"}<p>{\"b\"}</p>{\"\\n\"}</li>{\"\\n\"}</ul>"
            ),
            "{loose}"
        );
        // A nested list inside a tight item is not a paragraph, so it survives intact.
        let nested = commonmark("- a\n  - b\n");
        assert!(
            nested.contains(
                "<li>{\"a\"}{\"\\n\"}<ul>{\"\\n\"}<li>{\"b\"}</li>{\"\\n\"}</ul>{\"\\n\"}</li>"
            ),
            "{nested}"
        );
    }

    #[test]
    fn fenced_code_keeps_the_trailing_newline_inside_pre() {
        // mdast strips the closing fence's newline from `value`; inside a `<pre>` it is
        // significant, and every markdown pipeline puts it back.
        let out = commonmark("```js\nconst a = 1;\n```\n");
        assert!(out.contains("{\"const a = 1;\\n\"}"), "{out}");
        // ...but an empty fence stays empty rather than gaining a blank line.
        let empty = commonmark("```\n```\n");
        assert!(empty.contains("<pre><code /></pre>"), "{empty}");
    }

    #[test]
    fn urls_are_percent_normalized() {
        // `mdast-util-to-hast` runs every href/src through micromark's `normalizeUri`.
        let out = commonmark("[a](/caf\u{e9})\n");
        assert!(out.contains("href=\"/caf%C3%A9\""), "{out}");
        // An existing escape is left alone rather than double-encoded.
        assert!(commonmark("[a](/x%20y)\n").contains("href=\"/x%20y\""));
        // Reserved URL punctuation survives verbatim.
        assert!(
            commonmark("[a](https://e.com/p?q=1&r=2#f)\n")
                .contains("href=\"https://e.com/p?q=1&r=2#f\""),
            "{}",
            commonmark("[a](https://e.com/p?q=1&r=2#f)\n")
        );
    }

    #[test]
    fn gfm_elements_are_overridable_through_mdx_components() {
        // An app's `mdx-components.tsx` can restyle a GFM table or a task checkbox, the
        // same way it can restyle an `h1`.
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        std::fs::create_dir_all(root.join("app")).unwrap();
        std::fs::write(root.join("package.json"), "{}").unwrap();
        std::fs::write(
            root.join("mdx-components.tsx"),
            "export function useMDXComponents() { return {}; }\n",
        )
        .unwrap();
        let page = root.join("app/page.mdx");
        let out = compile_native(&page, "| a |\n| - |\n| 1 |\n\n- [x] t\n", Gfm::On)
            .unwrap()
            .jsx;
        for member in [
            "_components.table",
            "_components.thead",
            "_components.tbody",
            "_components.tr",
            "_components.th",
            "_components.td",
            "_components.input",
        ] {
            assert!(out.contains(member), "{member} missing from {out}");
        }
        assert!(
            out.contains("table: \"table\""),
            "defaults keep the intrinsic: {out}"
        );
        assert!(out.contains("del: \"del\""), "{out}");
    }
}
