//! The project's own import graph — what an app's ENTRY POINTS can actually reach.
//!
//! Island discovery for the Next app-router adapter is a filesystem walk
//! ([`crate::rsc::walk_project_modules`]): every `"use client"` module in the tree is
//! pinned into the client and SSR graphs so its client reference resolves. That
//! over-approximation is deliberate and is what makes the React Client Manifest
//! complete — but a filesystem walk has no idea whether a route can reach a file, so
//! without this module a `"use client"` file nobody imports (a leftover, an
//! `examples/` sketch, a `__tests__` helper) became a hard build dependency and its
//! unresolvable import failed the WHOLE build.
//!
//! This module supplies the missing fact, and only that fact: starting from the app's
//! real entry points, which project files are reachable, and which files name a
//! specifier no resolver can resolve. The adapter combines the two — a pinned island
//! that cannot be built is a hard error when a route can reach it, and is dropped
//! (loudly, naming the file and the specifier) when nothing can.
//!
//! Resolution mirrors [`crate::bundler`]'s: the same extensions, extension aliases
//! and tsconfig discovery, plus the caller's alias table (the adapter's `next/*`
//! shims), so a specifier this module cannot resolve is one the build cannot resolve
//! either. Node built-ins are external, never graph modules. Only files INSIDE the
//! project (and outside `node_modules`) become graph nodes: a dependency's internals
//! are the bundler's problem, not a reachability question.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};

use oxc_resolver::{ResolveOptions, Resolver, TsconfigDiscovery};

use crate::rsc::ModuleImport;

/// One module's dependencies as this graph sees them.
#[derive(Debug, Default, Clone)]
struct Node {
    /// Project files this module imports (canonical, inside the project).
    edges: Vec<PathBuf>,
    /// Specifiers the resolver rejected. A non-empty list is what makes this module —
    /// and everything that transitively imports it — unbuildable.
    unresolved: Vec<String>,
}

/// The import graph of the project's own files, rooted at a set of seeds.
#[derive(Debug, Default)]
pub struct ProjectImportGraph {
    nodes: HashMap<PathBuf, Node>,
}

/// Why a module cannot be built: the file that names the bad specifier, and the
/// specifier. `file` is the module itself for a direct failure, or a module it
/// transitively imports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnresolvedImport {
    pub file: PathBuf,
    pub specifier: String,
}

impl ProjectImportGraph {
    /// Walks the project import graph outward from `seeds`, resolving every
    /// non-type-only specifier of every project file it reaches.
    ///
    /// `sources` is the already-read module text keyed by canonical path (the
    /// adapter's project walk hands over what it read, so no file is read twice); a
    /// reached file that is not in it is read on demand, and one that cannot be read
    /// or is not a module (a stylesheet, an image) is a leaf.
    ///
    /// `aliases` is the build's `specifier -> path` alias table, applied exactly as
    /// the bundler applies it (exact match on a file, otherwise a path-prefix
    /// rewrite), so the adapter's `next/*` shims resolve here too.
    pub fn build(
        seeds: &[PathBuf],
        sources: &HashMap<PathBuf, String>,
        aliases: &[(String, String)],
    ) -> Self {
        let resolver = Resolver::new(resolve_options());
        let mut graph = ProjectImportGraph::default();
        let mut queue: VecDeque<PathBuf> = VecDeque::new();
        for seed in seeds {
            let seed = canonical(seed);
            if graph.nodes.contains_key(&seed) {
                continue;
            }
            graph.nodes.insert(seed.clone(), Node::default());
            queue.push_back(seed);
        }
        while let Some(file) = queue.pop_front() {
            let mut node = Node::default();
            for import in imports_of(&file, sources) {
                if import.type_only {
                    continue;
                }
                match resolve(&resolver, &file, &import.specifier, aliases) {
                    Resolution::External => {}
                    Resolution::Outside => {}
                    Resolution::Project(target) => {
                        if !graph.nodes.contains_key(&target) {
                            graph.nodes.insert(target.clone(), Node::default());
                            queue.push_back(target.clone());
                        }
                        node.edges.push(target);
                    }
                    Resolution::Unresolved => node.unresolved.push(import.specifier),
                }
            }
            // Nodes are inserted empty when first discovered (to dedupe the queue);
            // this is the real content.
            graph.nodes.insert(file, node);
        }
        graph
    }

    /// The project files reachable from `roots` (the roots themselves included), for
    /// roots that are part of this graph.
    pub fn reachable_from(&self, roots: &[PathBuf]) -> HashSet<PathBuf> {
        let mut seen: HashSet<PathBuf> = HashSet::new();
        let mut queue: VecDeque<PathBuf> = VecDeque::new();
        for root in roots {
            let root = canonical(root);
            if self.nodes.contains_key(&root) && seen.insert(root.clone()) {
                queue.push_back(root);
            }
        }
        while let Some(file) = queue.pop_front() {
            let Some(node) = self.nodes.get(&file) else {
                continue;
            };
            for edge in &node.edges {
                if seen.insert(edge.clone()) {
                    queue.push_back(edge.clone());
                }
            }
        }
        seen
    }

    /// The first unresolvable specifier `start` depends on, directly or through any
    /// module it transitively imports — i.e. the reason a build rooted at `start`
    /// would fail. `None` when everything it reaches resolves.
    ///
    /// Deterministic: the search is breadth-first from `start` and ties are broken by
    /// path, so the same project always reports the same failure.
    pub fn first_unresolved_from(&self, start: &Path) -> Option<UnresolvedImport> {
        let start = canonical(start);
        let mut seen: HashSet<PathBuf> = HashSet::from([start.clone()]);
        let mut queue: VecDeque<PathBuf> = VecDeque::from([start]);
        while let Some(file) = queue.pop_front() {
            let Some(node) = self.nodes.get(&file) else {
                continue;
            };
            if let Some(specifier) = node.unresolved.iter().min() {
                return Some(UnresolvedImport {
                    file,
                    specifier: specifier.clone(),
                });
            }
            let mut next: Vec<&PathBuf> = node.edges.iter().collect();
            next.sort();
            for edge in next {
                if seen.insert(edge.clone()) {
                    queue.push_back(edge.clone());
                }
            }
        }
        None
    }
}

/// What a specifier resolved to, from this graph's point of view.
enum Resolution {
    /// A Node built-in: external, left to the runtime.
    External,
    /// A real file the graph does not model (inside `node_modules`, or outside the
    /// project): resolvable, but not a reachability question.
    Outside,
    /// A project file.
    Project(PathBuf),
    /// Nothing matched — the build would fail here.
    Unresolved,
}

fn resolve(
    resolver: &Resolver,
    importer: &Path,
    specifier: &str,
    aliases: &[(String, String)],
) -> Resolution {
    if diffpack_default_loader::resolver_policy::is_external_specifier(specifier) {
        return Resolution::External;
    }
    // The loader query/fragment (`./a.css?url`) is not a filesystem concern.
    let path_specifier = diffpack_core::ResourceId::parse(specifier).path;
    // Aliases win before the resolver, exactly as the bundler orders them: an exact
    // match on a real file is the target; otherwise a path-prefix rewrite feeds the
    // normal resolver (so extensions and index files still apply).
    let mut aliased: Option<String> = None;
    for (from, target) in aliases {
        if from == &path_specifier {
            let target_path = Path::new(target);
            if target_path.is_file() {
                return classify(target_path);
            }
            aliased = Some(target.clone());
            break;
        }
        if let Some(rest) = path_specifier
            .strip_prefix(from.as_str())
            .and_then(|rest| rest.strip_prefix('/'))
        {
            aliased = Some(Path::new(target).join(rest).to_string_lossy().into_owned());
            break;
        }
    }
    let path_specifier = aliased.as_deref().unwrap_or(&path_specifier);
    // `resolve_file` (the importing FILE, not its directory) is the only entry point
    // `TsconfigDiscovery::Auto` applies to — the bundler calls it for the same reason.
    // Through `resolve` the tsconfig is never found, so every `paths` alias (`@/lib/x`,
    // the default in every `create-next-app` project) would look unresolvable and every
    // module behind one would look unreachable.
    match resolver.resolve_file(importer, path_specifier) {
        Ok(resolution) => classify(resolution.path()),
        Err(_) => Resolution::Unresolved,
    }
}

fn classify(path: &Path) -> Resolution {
    if path
        .components()
        .any(|component| component.as_os_str() == "node_modules")
    {
        return Resolution::Outside;
    }
    Resolution::Project(canonical(path))
}

fn imports_of(file: &Path, sources: &HashMap<PathBuf, String>) -> Vec<ModuleImport> {
    if let Some(source) = sources.get(file) {
        return crate::rsc::module_import_specifiers(file, source);
    }
    if !is_parseable_module(file) {
        return Vec::new();
    }
    match std::fs::read_to_string(file) {
        Ok(source) => crate::rsc::module_import_specifiers(file, &source),
        Err(_) => Vec::new(),
    }
}

/// Whether the graph parses this file for imports. A stylesheet or an image is a
/// legitimate leaf, not a module with dependencies.
fn is_parseable_module(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|value| value.to_str()),
        Some("ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs" | "mts" | "cts")
    )
}

fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// The resolver configuration, kept in step with [`crate::bundler`]'s: the same
/// extensions, the same TypeScript extension aliases, the same automatic tsconfig
/// discovery (so a `paths` alias like `@/components/x` resolves here too).
/// Conditions are fixed rather than per-environment: the island set MUST be identical
/// in the client, react-server and SSR passes, or their manifests disagree on which
/// modules exist.
fn resolve_options() -> ResolveOptions {
    ResolveOptions {
        tsconfig: Some(TsconfigDiscovery::Auto),
        extensions: [
            ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".json", ".mdx", ".md",
        ]
        .into_iter()
        .map(String::from)
        .collect(),
        extension_alias: vec![
            (
                ".js".into(),
                vec![".ts".into(), ".tsx".into(), ".js".into(), ".jsx".into()],
            ),
            (".mjs".into(), vec![".mts".into(), ".mjs".into()]),
            (".cjs".into(), vec![".cts".into(), ".cjs".into()]),
        ],
        condition_names: vec![
            "import".into(),
            "module".into(),
            "browser".into(),
            "default".into(),
        ],
        main_fields: vec!["browser".into(), "module".into(), "main".into()],
        ..ResolveOptions::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("diffpack-project-graph-{name}"));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(root: &Path, relative: &str, source: &str) -> PathBuf {
        let path = root.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, source).unwrap();
        std::fs::canonicalize(&path).unwrap()
    }

    #[test]
    fn reachability_follows_static_dynamic_and_require_edges() {
        let root = scratch("edges");
        let page = write(
            &root,
            "app/page.tsx",
            "import A from \"../lib/a\";\nconst B = () => import(\"../lib/b\");\nexport default function P() { return [A, B]; }\n",
        );
        let a = write(
            &root,
            "lib/a.ts",
            "const C = require(\"./c\");\nexport default C;\n",
        );
        let b = write(&root, "lib/b.ts", "export default 2;\n");
        let c = write(&root, "lib/c.ts", "export default 3;\n");
        let orphan = write(&root, "lib/orphan.ts", "export default 4;\n");

        let graph = ProjectImportGraph::build(std::slice::from_ref(&page), &HashMap::new(), &[]);
        let reachable = graph.reachable_from(std::slice::from_ref(&page));
        for expected in [&page, &a, &b, &c] {
            assert!(
                reachable.contains(expected),
                "{} not reachable",
                expected.display()
            );
        }
        assert!(
            !reachable.contains(&orphan),
            "an unimported module is not reachable"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_type_only_import_is_never_resolved() {
        // `import type` is erased before the bundler sees it, so it may name a `.d.ts`
        // no resolver finds. Treating it as a dependency would report a build failure
        // that cannot happen.
        let root = scratch("type-only");
        let module = write(
            &root,
            "lib/x.ts",
            "import type { Shape } from \"./shape\";\nexport const x: Shape = 1 as never;\n",
        );
        let graph = ProjectImportGraph::build(std::slice::from_ref(&module), &HashMap::new(), &[]);
        assert_eq!(graph.first_unresolved_from(&module), None);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn an_unresolvable_specifier_is_reported_through_the_whole_chain() {
        let root = scratch("unresolved");
        let entry = write(&root, "lib/entry.ts", "export { mid } from \"./mid\";\n");
        let mid = write(
            &root,
            "lib/mid.ts",
            "export { deep as mid } from \"./missing\";\n",
        );
        let graph = ProjectImportGraph::build(std::slice::from_ref(&entry), &HashMap::new(), &[]);
        assert_eq!(
            graph.first_unresolved_from(&entry),
            Some(UnresolvedImport {
                file: mid,
                specifier: "./missing".to_string()
            })
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn node_builtins_and_dependencies_are_not_unresolved() {
        let root = scratch("externals");
        let module = write(
            &root,
            "lib/x.ts",
            "import { readFileSync } from \"node:fs\";\nimport path from \"path\";\nexport const x = [readFileSync, path];\n",
        );
        let graph = ProjectImportGraph::build(std::slice::from_ref(&module), &HashMap::new(), &[]);
        assert_eq!(graph.first_unresolved_from(&module), None);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_tsconfig_paths_alias_resolves() {
        // REGRESSION. `create-next-app` writes `"paths": { "@/*": ["./*"] }` and every
        // route imports through it. Resolving from the importer's DIRECTORY never finds
        // the tsconfig (oxc_resolver applies `TsconfigDiscovery::Auto` only to
        // `resolve_file`), so every `@/…` specifier looked unresolvable, every module
        // behind one looked unreachable, and a live client component was dropped.
        let root = scratch("tsconfig-paths");
        std::fs::write(
            root.join("tsconfig.json"),
            "{ \"compilerOptions\": { \"paths\": { \"@/*\": [\"./*\"] } } }\n",
        )
        .unwrap();
        let layout = write(
            &root,
            "app/layout.tsx",
            "import Client from \"@/lib/client-layout\";\nexport default function L() { return Client; }\n",
        );
        let client = write(
            &root,
            "lib/client-layout.tsx",
            "export default function C() { return null; }\n",
        );
        let graph = ProjectImportGraph::build(std::slice::from_ref(&layout), &HashMap::new(), &[]);
        assert_eq!(graph.first_unresolved_from(&layout), None);
        assert!(
            graph
                .reachable_from(std::slice::from_ref(&layout))
                .contains(&client),
            "a module imported through a tsconfig `paths` alias is reachable",
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn an_alias_resolves_exactly_as_the_bundler_resolves_it() {
        let root = scratch("alias");
        let shim = write(
            &root,
            "shims/link.tsx",
            "export default function Link() { return null; }\n",
        );
        let module = write(
            &root,
            "lib/nav.tsx",
            "import Link from \"next/link\";\nexport default Link;\n",
        );
        let aliases = vec![("next/link".to_string(), shim.to_string_lossy().into_owned())];
        let graph =
            ProjectImportGraph::build(std::slice::from_ref(&module), &HashMap::new(), &aliases);
        assert_eq!(graph.first_unresolved_from(&module), None);
        assert!(graph.reachable_from(&[module]).contains(&shim));
        std::fs::remove_dir_all(&root).ok();
    }
}
