//! Native generation of TanStack Router's `routeTree.gen.ts`.
//!
//! TanStack Start apps import a generated `src/routeTree.gen.ts` — the assembled,
//! typed route tree — from `router.tsx`. Upstream that file is produced by
//! TanStack Router's **Vite plugin**: it watches `src/routes/`, classifies each
//! filename against the file-route convention, and emits the tree. That made the
//! Vite tooling diffpack claims to replace a mandatory producer of a build-path
//! INPUT, and made adding/renaming/removing a route impossible without it.
//!
//! This module replaces that plugin natively, in Rust, with no Vite/Babel/Node:
//! it walks `<root>/src/routes` recursively, parses every filename into the
//! file-route convention model (index, flat dot-nesting, directory nesting,
//! pathless `_layout` routes, dynamic `$param` segments, escaped `[.]` literals,
//! and the trailing-`_` nesting opt-out), builds the parent/child route graph,
//! and emits a runtime-complete `routeTree.gen.ts`.
//!
//! Only the RUNTIME structure is emitted (the per-route `import`, the
//! `Import.update({ id, path, getParentRoute })` chain, the `_addFileChildren`
//! assembly, and the exported `routeTree`). The upstream `declare module` /
//! `interface` blocks are type-only — stripped by the TS transform at build — so
//! they are intentionally omitted; the file is `@ts-nocheck` and its runtime
//! object is identical in shape to the upstream one.
//!
//! Generation is a build-emit step, off the incremental transform hot path
//! (mirroring native manifest generation): a leaf edit still re-transforms
//! exactly one module and the low-memory thesis is untouched.
//!
//! Anything the classifier cannot map is a HARD ERROR naming the file — never a
//! silent drop or a wrong placeholder id.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

/// The generated file, relative to the app source directory.
pub const ROUTE_TREE_FILE: &str = "routeTree.gen.ts";

/// The route-file extensions the convention treats as routes.
const ROUTE_EXTENSIONS: [&str; 4] = ["tsx", "ts", "jsx", "js"];

/// A single parsed route, in the file-route convention model. `full_id` is the
/// canonical route id (identical to the `createFileRoute('<id>')` literal the
/// route source carries and to TanStack's `FileRoutesById` key), which makes the
/// model directly comparable to the upstream reference.
#[derive(Debug, Clone)]
pub struct Route {
    /// Module specifier used in the generated import, e.g. `./routes/users.$userId`
    /// (verbatim filename, keeping `[.]` escapes and `$` params).
    pub import_path: String,
    /// The route file on disk.
    pub source_file: PathBuf,
    /// Convention segments (dot/slash separated, `[.]` unescaped, terminal
    /// `index` removed), e.g. `["posts_", "$postId", "deep"]`.
    pub segments: Vec<String>,
    /// Whether the leaf segment was `index`.
    pub is_index: bool,
    /// Canonical full route id, e.g. `/users/$userId`, `/`, `/users/`.
    pub full_id: String,
    /// PascalCase base for the generated variable names (`Users`, `UsersUserId`).
    pub var_base: String,
    /// Index (into the route vec) of this route's parent, or `None` for `__root__`.
    pub parent: Option<usize>,
    /// The `path` field emitted in `.update(...)`; `None` for a pathless layout.
    pub path: Option<String>,
    /// The `id` field emitted in `.update(...)` (own id relative to the parent).
    pub local_id: String,
}

/// The assembled route graph: a flat route list plus the root children.
#[derive(Debug, Clone)]
pub struct RouteTree {
    pub routes: Vec<Route>,
    /// Import specifier for the root route (`createRootRoute`), under the routes
    /// directory's name (e.g. `./routes/__root` or `./app/__root`).
    root_import: String,
    /// Indices (into `routes`) of the direct children of `__root__`.
    root_children: Vec<usize>,
    /// `children[i]` = indices of the direct children of `routes[i]`.
    children: Vec<Vec<usize>>,
}

impl RouteTree {
    /// The semantic identity of the tree: the set of
    /// `(full_id, path, parent_full_id)` triples. `__root__` is the parent id for
    /// a top-level route. This is exactly what "semantically equivalent to the
    /// reference tree" means — same route ids, paths, and parent edges — and is
    /// independent of variable names or emission order.
    pub fn semantic_set(&self) -> BTreeSet<(String, Option<String>, String)> {
        self.routes
            .iter()
            .map(|route| {
                let parent_id = match route.parent {
                    Some(index) => self.routes[index].full_id.clone(),
                    None => ROOT_ID.to_string(),
                };
                (route.full_id.clone(), route.path.clone(), parent_id)
            })
            .collect()
    }
}

/// The canonical id of the framework root route.
pub const ROOT_ID: &str = "__root__";

/// Walk `<root>/src/routes`, generate the route tree, and write it to
/// `<root>/src/routeTree.gen.ts`. Returns `Ok(None)` when the app has no
/// `src/routes` directory (a non-file-routed bundle), so callers can no-op. On a
/// routed app, returns the number of routes written.
///
/// This is the build-emit entry point wired into `build-app` and `dev` startup:
/// it makes diffpack — not TanStack's Vite plugin — the producer of the route
/// tree, closing the last build-path dependency on Vite tooling.
pub fn generate_for_project(project_root: &Path) -> Result<Option<usize>, String> {
    let source_dir = source_dir(project_root);
    // The routes directory is `router.routesDirectory` (relative to the source
    // directory), defaulting to `routes` — TanStack Start apps that set it (e.g.
    // `routesDirectory: "app"`) keep their routes elsewhere, and hardcoding
    // `routes` silently produced no route tree for them.
    let routes_name =
        crate::config::vite_config_string(project_root, "routesDirectory")
            .unwrap_or_else(|| DEFAULT_ROUTES_DIR.to_string());
    let routes_dir = source_dir.join(&routes_name);
    if !routes_dir.is_dir() {
        return Ok(None);
    }
    let ignore = RouteIgnore::parse(
        crate::config::vite_config_string(project_root, "routeFileIgnorePattern").as_deref(),
    )?;
    let tree = generate_from_routes_dir_with(&routes_dir, &routes_name, &ignore)?;
    let source = tree.emit();
    let output = source_dir.join(ROUTE_TREE_FILE);
    write_if_changed(&output, source.as_bytes())?;
    Ok(Some(tree.routes.len()))
}

/// The default `router.routesDirectory`, relative to the source directory.
const DEFAULT_ROUTES_DIR: &str = "routes";

/// TanStack Router's non-route file filter: the built-in `-`/`.` prefix
/// colocation convention plus the configurable `routeFileIgnorePattern`.
///
/// `routeFileIgnorePattern` is a JavaScript regular expression tested (unanchored)
/// against each route file path. Diffpack does not embed a regex engine, so it
/// supports the literal-alternation form real apps use (`(_components|api)`,
/// `admin`, `foo|bar`) and hard-errors on anything using regex metacharacters
/// rather than silently mis-matching — a loud, specific pointer at the next gap.
#[derive(Debug)]
struct RouteIgnore {
    /// Literal substrings; a path matching any is not a route (unanchored, exactly
    /// as `new RegExp(pattern).test(path)` would behave for these forms).
    substrings: Vec<String>,
}

impl RouteIgnore {
    /// No configured pattern: only the built-in `-`/`.` prefix convention applies.
    fn none() -> Self {
        Self { substrings: Vec::new() }
    }

    fn parse(pattern: Option<&str>) -> Result<Self, String> {
        let Some(pattern) = pattern else {
            return Ok(Self::none());
        };
        // Strip one optional wrapping group so `(_components|api)` and
        // `_components|api` are handled identically, then split the alternation.
        let trimmed = pattern.trim();
        let inner = trimmed
            .strip_prefix('(')
            .and_then(|body| body.strip_suffix(')'))
            .unwrap_or(trimmed);
        let mut substrings = Vec::new();
        for alternative in inner.split('|') {
            let alternative = alternative.trim();
            if alternative.is_empty() {
                continue;
            }
            let literal = alternative.chars().all(|character| {
                character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '/')
            });
            if !literal {
                return Err(format!(
                    "routeFileIgnorePattern {pattern:?} uses regex features Diffpack does not yet \
                     support: only literal alternation like \"(_components|api)\" is handled. \
                     Extend RouteIgnore in src/route_tree.rs to support this pattern."
                ));
            }
            substrings.push(alternative.to_string());
        }
        Ok(Self { substrings })
    }

    /// Whether a directory/file entry named `name` at route-relative `rel` is a
    /// non-route file. `name` catches the `-`/`.` prefix convention (checked per
    /// path component); `rel` is matched against the ignore substrings.
    fn excludes(&self, name: &str, rel: &str) -> bool {
        name.starts_with('-')
            || name.starts_with('.')
            || self.substrings.iter().any(|needle| rel.contains(needle))
    }
}

/// The app source directory (`src` if present, else the project root itself),
/// mirroring `config`/`dev_server`'s `srcDirectory` default.
fn source_dir(project_root: &Path) -> PathBuf {
    let src = project_root.join("src");
    if src.is_dir() {
        src
    } else {
        project_root.to_path_buf()
    }
}

/// Parse every route file under `routes_dir` (the default `routes` directory,
/// with no ignore pattern) and assemble the route graph. Kept for the tests and
/// callers that use the convention default; [`generate_from_routes_dir_with`] is
/// the configurable core.
pub fn generate_from_routes_dir(routes_dir: &Path) -> Result<RouteTree, String> {
    generate_from_routes_dir_with(routes_dir, DEFAULT_ROUTES_DIR, &RouteIgnore::none())
}

/// Parse every route file under `routes_dir` and assemble the route graph.
/// `import_prefix` is the routes directory's name relative to the source
/// directory (`routes`, `app`, ...) and becomes the `./<prefix>/...` specifier in
/// the generated imports; `ignore` filters non-route files.
fn generate_from_routes_dir_with(
    routes_dir: &Path,
    import_prefix: &str,
    ignore: &RouteIgnore,
) -> Result<RouteTree, String> {
    let mut files = Vec::new();
    collect_route_files(routes_dir, routes_dir, ignore, &mut files)?;
    files.sort();

    let mut routes: Vec<Route> = Vec::new();
    let mut has_root = false;
    for relative in &files {
        // `__root` is the framework root route (`createRootRoute`), not a
        // file-convention route: it becomes `rootRouteImport`, the tree's apex.
        if relative == "__root" {
            has_root = true;
            continue;
        }
        routes.push(parse_route(routes_dir, import_prefix, relative)?);
    }
    if !has_root {
        return Err(format!(
            "cannot generate route tree: no root route (__root.tsx) found under {}",
            routes_dir.display()
        ));
    }

    assign_variable_names(&mut routes);
    resolve_parents_and_paths(&mut routes)?;

    let (root_children, children) = build_child_lists(&routes);
    Ok(RouteTree {
        routes,
        root_import: format!("./{import_prefix}/__root"),
        root_children,
        children,
    })
}

/// Recursively collect route files under `dir`, as paths relative to `root` with
/// the extension stripped (e.g. `api/users.$userId`, `_pathlessLayout`). Non-route
/// files filtered by `ignore` (the `-`/`.` colocation convention and the
/// configured ignore pattern) are skipped before classification. A non-route
/// extension under the routes tree is otherwise a hard error rather than a silent
/// skip, so an unexpected file cannot vanish from the build.
fn collect_route_files(
    root: &Path,
    dir: &Path,
    ignore: &RouteIgnore,
    out: &mut Vec<String>,
) -> Result<(), String> {
    let entries = fs::read_dir(dir)
        .map_err(|error| format!("cannot read routes directory {}: {error}", dir.display()))?;
    for entry in entries {
        let entry =
            entry.map_err(|error| format!("cannot read {}: {error}", dir.display()))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|error| format!("cannot stat {}: {error}", path.display()))?;
        let name = entry.file_name().to_string_lossy().to_string();
        let relative = path
            .strip_prefix(root)
            .map_err(|error| format!("cannot relativize {}: {error}", path.display()))?;
        let relative = relative.to_string_lossy().replace('\\', "/");
        // A colocation dir/file (`-components`, `.DS_Store`) or an ignore-pattern
        // match (`_components`, `api`) is not a route: skip it (and, for a
        // directory, everything under it) before it can trip classification.
        if ignore.excludes(&name, &relative) {
            continue;
        }
        if file_type.is_dir() {
            collect_route_files(root, &path, ignore, out)?;
            continue;
        }
        let extension = path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("");
        // Only the configured route extensions are routes. A colocated asset
        // (`favicon.ico`, a `.css`, an image, a `.md`) in the routes directory is
        // not a route and is skipped, exactly as TanStack's generator ignores it —
        // real apps colocate such files next to the routes that use them.
        if !ROUTE_EXTENSIONS.contains(&extension) {
            continue;
        }
        // Relative path without extension, forward-slash normalized.
        let mut stem = relative;
        let dot = stem.len() - extension.len() - 1;
        stem.truncate(dot);
        out.push(stem);
    }
    Ok(())
}

/// Parse one route file (relative path, no extension) into a [`Route`], deriving
/// `full_id`. Parent/path/local_id are filled by a later pass once all routes are
/// known.
fn parse_route(routes_dir: &Path, import_prefix: &str, relative: &str) -> Result<Route, String> {
    let source_file_display = routes_dir.join(relative);
    // `segments` already excludes a terminal `index` (folded into full_id), so
    // downstream parent matching is uniform.
    let (segments, is_index) = classify_filename(relative).map_err(|reason| {
        format!(
            "cannot classify route filename {}: {reason}",
            source_file_display.display()
        )
    })?;

    let full_id = full_id_of(&segments, is_index);

    // The import specifier keeps the ORIGINAL filename (escapes and params
    // intact) under the routes directory's name (`routes`, `app`, ...), which is
    // what the emitted `import ... from '<here>'` must resolve relative to the
    // generated route tree in the source directory.
    let import_path = format!("./{import_prefix}/{relative}");

    Ok(Route {
        import_path,
        source_file: source_file_display,
        segments,
        is_index,
        full_id,
        var_base: String::new(),
        parent: None,
        path: None,
        local_id: String::new(),
    })
}

/// Split a route filename into convention segments and detect a terminal
/// `index`. Both `/` (directory nesting) and `.` (flat dot-nesting) separate
/// segments; `[x]` escapes the literal char `x` (so `[.]` is a literal dot inside
/// a segment, not a separator). An empty segment or an unbalanced `[`/`]` is a
/// hard error — the classifier never guesses.
fn classify_filename(relative: &str) -> Result<(Vec<String>, bool), String> {
    let mut segments: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut chars = relative.chars().peekable();
    while let Some(character) = chars.next() {
        match character {
            '/' | '.' => {
                segments.push(std::mem::take(&mut current));
            }
            '[' => {
                let escaped = chars.next().ok_or_else(|| {
                    "unterminated `[` escape (expected `[<char>]`)".to_string()
                })?;
                match chars.next() {
                    Some(']') => current.push(escaped),
                    Some(other) => {
                        return Err(format!(
                            "malformed `[` escape: expected `]` after `[{escaped}`, found `{other}`"
                        ));
                    }
                    None => {
                        return Err("unterminated `[` escape (expected `]`)".to_string());
                    }
                }
            }
            ']' => {
                return Err("unbalanced `]` outside a `[<char>]` escape".to_string());
            }
            other => current.push(other),
        }
    }
    segments.push(current);

    for segment in &segments {
        if segment.is_empty() {
            return Err(
                "empty path segment (a stray `.`/`/` or a doubled separator)".to_string(),
            );
        }
    }

    // A terminal `index` segment is the directory's index route: it contributes
    // no path segment and folds into a trailing slash on the id.
    let is_index = segments.last().map(|value| value == "index").unwrap_or(false);
    if is_index {
        segments.pop();
    }
    Ok((segments, is_index))
}

/// The canonical full route id for a set of (index-stripped) segments.
fn full_id_of(segments: &[String], is_index: bool) -> String {
    if segments.is_empty() {
        // The bare index route (`index.tsx`) is the root index, `/`.
        return "/".to_string();
    }
    let joined = segments.join("/");
    if is_index {
        format!("/{joined}/")
    } else {
        format!("/{joined}")
    }
}

/// Resolve each route's parent (by longest matching id prefix, honoring the
/// trailing-`_` opt-out and flattening a missing intermediate to root), then
/// derive its `path` and local `id`.
fn resolve_parents_and_paths(routes: &mut [Route]) -> Result<(), String> {
    // full_id -> route index, for parent lookup.
    let by_full_id: BTreeMap<String, usize> = routes
        .iter()
        .enumerate()
        .map(|(index, route)| (route.full_id.clone(), index))
        .collect();

    for index in 0..routes.len() {
        // The segments the parent would "own": all but this route's own leaf
        // segment. An index route's own leaf is the (already stripped) `index`,
        // so for an index route the parent owns ALL remaining segments.
        let route_segments = routes[index].segments.clone();
        let is_index = routes[index].is_index;

        let parent_segment_count = if is_index {
            route_segments.len()
        } else {
            route_segments.len().saturating_sub(1)
        };
        let parent_segments = &route_segments[..parent_segment_count];

        let parent = if parent_segments.is_empty() {
            None
        } else {
            let target = format!("/{}", parent_segments.join("/"));
            match by_full_id.get(&target).copied() {
                Some(parent_index) => Some(parent_index),
                None => {
                    // A missing intermediate normally flattens to root (e.g. an
                    // `api/users` route with no `api` layout keeps its full path).
                    // But a MISSING pathless layout is a real broken tree — the
                    // `_layout` file the child nests under does not exist — and is a
                    // hard error, since silently reparenting to root would drop the
                    // layout and change routing.
                    let last = parent_segments.last().map(String::as_str).unwrap_or("");
                    if last.starts_with('_') {
                        return Err(format!(
                            "cannot resolve route parent for {}: pathless layout route {:?} (file {}) does not exist",
                            routes[index].full_id,
                            target,
                            routes[index].source_file.display()
                        ));
                    }
                    None
                }
            }
        };

        // The route's own segments = everything the parent did not own.
        let consumed = match parent {
            Some(parent_index) => routes[parent_index].segments.len(),
            None => 0,
        };
        let own_segments = &route_segments[consumed..];

        let (path, local_id) = derive_path_and_id(own_segments, is_index);
        routes[index].parent = parent;
        routes[index].path = path;
        routes[index].local_id = local_id;
    }
    Ok(())
}

/// Derive a route's `path` (routing contribution) and local `id` from its own
/// segments (those beyond its parent). Pathless `_`-prefixed segments contribute
/// no path; a trailing `_` (nesting opt-out) is stripped from the path but kept
/// in the id; a terminal `index` is a `/` path.
fn derive_path_and_id(own_segments: &[String], is_index: bool) -> (Option<String>, String) {
    if is_index {
        // An index route's own id is `/` and its path is `/`.
        return (Some("/".to_string()), "/".to_string());
    }

    // The local id keeps every own segment verbatim (params, escapes, `_`).
    let local_id = format!("/{}", own_segments.join("/"));

    // The path drops pathless `_`-prefixed layout segments entirely and strips a
    // trailing `_` opt-out from the remaining segments.
    let path_segments: Vec<String> = own_segments
        .iter()
        .filter(|segment| !segment.starts_with('_'))
        .map(|segment| segment.strip_suffix('_').unwrap_or(segment).to_string())
        .collect();
    let path = if path_segments.is_empty() {
        // Every own segment was a pathless layout: this route contributes no path.
        None
    } else {
        Some(format!("/{}", path_segments.join("/")))
    };

    (path, local_id)
}

/// Assign a unique PascalCase base name to each route from its `full_id`.
fn assign_variable_names(routes: &mut [Route]) {
    let mut used: BTreeSet<String> = BTreeSet::new();
    for route in routes.iter_mut() {
        let mut base = pascal_base(&route.full_id);
        if base.is_empty() {
            base = "Route".to_string();
        }
        // Disambiguate any (rare) sanitization collision deterministically.
        let mut candidate = base.clone();
        let mut counter = 1;
        while !used.insert(candidate.clone()) {
            counter += 1;
            candidate = format!("{base}{counter}");
        }
        route.var_base = candidate;
    }
}

/// Build a PascalCase identifier base from a full route id: each `/`-segment is
/// title-cased (splitting on `-`, `.`, `$`), leading/trailing `_` stripped, and a
/// trailing slash (index) rendered as `Index`.
fn pascal_base(full_id: &str) -> String {
    if full_id == "/" {
        return "Index".to_string();
    }
    let is_index = full_id.ends_with('/');
    let mut out = String::new();
    for segment in full_id.split('/').filter(|value| !value.is_empty()) {
        let trimmed = segment.trim_start_matches('_').trim_end_matches('_');
        for word in trimmed.split(['-', '.', '$']).filter(|value| !value.is_empty()) {
            let mut word_chars = word.chars();
            if let Some(first) = word_chars.next() {
                out.extend(first.to_uppercase());
                out.push_str(word_chars.as_str());
            }
        }
    }
    if is_index {
        out.push_str("Index");
    }
    out
}

/// Compute the direct-children lists (root children + per-route children),
/// each sorted by `full_id` for deterministic emission.
fn build_child_lists(routes: &[Route]) -> (Vec<usize>, Vec<Vec<usize>>) {
    let mut root_children: Vec<usize> = Vec::new();
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); routes.len()];
    let mut order: Vec<usize> = (0..routes.len()).collect();
    order.sort_by(|&a, &b| routes[a].full_id.cmp(&routes[b].full_id));
    for index in order {
        match routes[index].parent {
            Some(parent) => children[parent].push(index),
            None => root_children.push(index),
        }
    }
    (root_children, children)
}

impl RouteTree {
    /// Emit the runtime-complete `routeTree.gen.ts` source.
    pub fn emit(&self) -> String {
        let mut out = String::new();
        out.push_str("/* eslint-disable */\n\n");
        out.push_str("// @ts-nocheck\n\n");
        out.push_str(
            "// This file was automatically generated by diffpack's native route-tree\n\
             // generator (replacing TanStack Router's Vite plugin). Do not edit.\n\n",
        );

        // Imports: the root first, then every route.
        out.push_str(&format!(
            "import {{ Route as rootRouteImport }} from '{}'\n",
            self.root_import
        ));
        for route in &self.routes {
            out.push_str(&format!(
                "import {{ Route as {base}RouteImport }} from '{path}'\n",
                base = route.var_base,
                path = route.import_path,
            ));
        }
        out.push('\n');

        // The `.update(...)` const for every route. `getParentRoute` is a thunk,
        // so referencing another route const before its definition is safe.
        for (index, route) in self.routes.iter().enumerate() {
            let parent_var = match route.parent {
                Some(parent) => format!("{}Route", self.routes[parent].var_base),
                None => "rootRouteImport".to_string(),
            };
            out.push_str(&format!(
                "const {base}Route = {base}RouteImport.update({{\n  id: '{id}',\n",
                base = route.var_base,
                id = escape_single(&route.local_id),
            ));
            if let Some(path) = &route.path {
                out.push_str(&format!("  path: '{}',\n", escape_single(path)));
            }
            out.push_str(&format!(
                "  getParentRoute: () => {parent_var},\n}} as any)\n"
            ));
            let _ = index;
        }
        out.push('\n');

        // Child assembly. A parent's children object references a child's
        // `WithChildren` variant iff that child itself has children, so children
        // must be emitted before their parents: deepest routes first.
        let mut parents_with_children: Vec<usize> = (0..self.routes.len())
            .filter(|&index| !self.children[index].is_empty())
            .collect();
        parents_with_children.sort_by(|&a, &b| {
            let depth = |index: usize| self.routes[index].full_id.matches('/').count();
            depth(b)
                .cmp(&depth(a))
                .then_with(|| self.routes[a].full_id.cmp(&self.routes[b].full_id))
        });

        for &parent in &parents_with_children {
            let base = &self.routes[parent].var_base;
            out.push_str(&format!("const {base}RouteChildren = {{\n"));
            for &child in &self.children[parent] {
                out.push_str(&format!(
                    "  {key}: {value},\n",
                    key = child_key(&self.routes[child]),
                    value = self.child_ref(child),
                ));
            }
            out.push_str("}\n");
            out.push_str(&format!(
                "const {base}RouteWithChildren = {base}Route._addFileChildren({base}RouteChildren)\n\n",
            ));
        }

        // Root assembly and the exported tree.
        out.push_str("const rootRouteChildren = {\n");
        for &child in &self.root_children {
            out.push_str(&format!(
                "  {key}: {value},\n",
                key = child_key(&self.routes[child]),
                value = self.child_ref(child),
            ));
        }
        out.push_str("}\n");
        out.push_str(
            "export const routeTree = rootRouteImport._addFileChildren(rootRouteChildren)\n",
        );
        out
    }

    /// The variable a parent's children object should reference for `child`: the
    /// `WithChildren` variant if the child has its own children, else the plain
    /// route const.
    fn child_ref(&self, child: usize) -> String {
        let base = &self.routes[child].var_base;
        if self.children[child].is_empty() {
            format!("{base}Route")
        } else {
            format!("{base}RouteWithChildren")
        }
    }
}

/// The object KEY used for a child inside a `...RouteChildren` object — always the
/// plain `<Base>Route` identifier (the value may be the `WithChildren` variant).
fn child_key(route: &Route) -> String {
    format!("{}Route", route.var_base)
}

/// Escape a single-quoted JS string literal body.
fn escape_single(value: &str) -> String {
    value.replace('\\', "\\\\").replace('\'', "\\'")
}

/// Write `bytes` to `path` only if the current contents differ, so a redundant
/// regeneration does not churn mtimes (and cannot self-trigger the dev watcher).
fn write_if_changed(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Ok(existing) = fs::read(path)
        && existing == bytes
    {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    }
    fs::write(path, bytes)
        .map_err(|error| format!("cannot write {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn fixture_routes_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("integration/tanstack-start-reference/src/routes")
    }

    fn reference_gen_ts() -> String {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("integration/tanstack-start-reference/src/routeTree.gen.ts");
        fs::read_to_string(path).expect("reference routeTree.gen.ts must exist")
    }

    /// A parsed `.update(...)` const from the committed reference.
    struct RefUpdate {
        local_id: String,
        path: Option<String>,
        parent_var: String,
    }

    /// Parse the reference file's RUNTIME section (`const XRoute =
    /// XRouteImport.update({...})`) into a var -> update map, independent of
    /// diffpack's own generator. This reads the actual committed file, so the
    /// equivalence test cannot be gamed by hardcoding.
    fn parse_reference_updates(source: &str) -> HashMap<String, RefUpdate> {
        let mut updates = HashMap::new();
        let lines: Vec<&str> = source.lines().collect();
        let mut index = 0;
        while index < lines.len() {
            let line = lines[index].trim_start();
            if let Some(var) = parse_update_header(line) {
                let mut local_id = None;
                let mut path = None;
                let mut parent_var = None;
                let mut cursor = index + 1;
                while cursor < lines.len() {
                    let body = lines[cursor].trim();
                    if body.starts_with("} as any)") || body == "} as any)" {
                        break;
                    }
                    if let Some(value) = body.strip_prefix("id: '") {
                        local_id = value.strip_suffix("',").map(str::to_string);
                    } else if let Some(value) = body.strip_prefix("path: '") {
                        path = value.strip_suffix("',").map(str::to_string);
                    } else if let Some(value) = body.strip_prefix("getParentRoute: () => ") {
                        parent_var = Some(value.trim_end_matches(',').to_string());
                    }
                    cursor += 1;
                }
                updates.insert(
                    var,
                    RefUpdate {
                        local_id: local_id.expect("update has an id"),
                        path,
                        parent_var: parent_var.expect("update has a parent"),
                    },
                );
                index = cursor;
            }
            index += 1;
        }
        updates
    }

    /// `const <Var>Route = <Var>RouteImport.update({` (possibly wrapped so the
    /// `.update({` lands on a later line). Returns the assigned variable name.
    fn parse_update_header(line: &str) -> Option<String> {
        let rest = line.strip_prefix("const ")?;
        // Stop the name at whitespace, `=`, or a `:` type annotation.
        let var = rest.split([' ', '=', ':']).next()?.to_string();
        if var.is_empty() || var.ends_with("Children") {
            return None;
        }
        // Only accept lines that introduce an `.update(` call (directly or via
        // the wrapped `X =\n  XImport.update({` form both appear in the file).
        if line.contains(".update({") || line.trim_end().ends_with('=') {
            Some(var)
        } else {
            None
        }
    }

    /// Resolve the full id of a reference var by walking its parent chain.
    fn reference_full_id(var: &str, updates: &HashMap<String, RefUpdate>) -> String {
        if var == "rootRouteImport" {
            return String::new();
        }
        let update = &updates[var];
        let parent_full = reference_full_id(&update.parent_var, updates);
        format!("{parent_full}{}", update.local_id)
    }

    /// The reference tree's semantic set — same shape as
    /// [`RouteTree::semantic_set`] — parsed from the committed file.
    fn reference_semantic_set(source: &str) -> BTreeSet<(String, Option<String>, String)> {
        let updates = parse_reference_updates(source);
        updates
            .iter()
            .map(|(var, update)| {
                let full_id = reference_full_id(var, &updates);
                let parent_label = if update.parent_var == "rootRouteImport" {
                    ROOT_ID.to_string()
                } else {
                    reference_full_id(&update.parent_var, &updates)
                };
                (full_id, update.path.clone(), parent_label)
            })
            .collect()
    }

    #[test]
    fn generated_tree_is_semantically_equivalent_to_the_reference() {
        let tree = generate_from_routes_dir(&fixture_routes_dir())
            .expect("fixture routes must classify");
        let generated = tree.semantic_set();
        let reference = reference_semantic_set(&reference_gen_ts());

        assert_eq!(
            generated.len(),
            reference.len(),
            "route count mismatch: generated {} vs reference {}",
            generated.len(),
            reference.len()
        );
        // Compare the full (id, path, parent-edge) sets — the semantic identity
        // of the tree, independent of variable names or emission order.
        let only_generated: Vec<_> = generated.difference(&reference).collect();
        let only_reference: Vec<_> = reference.difference(&generated).collect();
        assert!(
            only_generated.is_empty() && only_reference.is_empty(),
            "semantic mismatch\n  only in generated: {only_generated:?}\n  only in reference: {only_reference:?}"
        );
    }

    #[test]
    fn generated_tree_covers_every_reference_route_id_path_and_parent() {
        // Spot-check the load-bearing convention cases explicitly so a regression
        // names the exact route, not just a set diff.
        let tree = generate_from_routes_dir(&fixture_routes_dir()).unwrap();
        let by_id: HashMap<&str, &Route> =
            tree.routes.iter().map(|r| (r.full_id.as_str(), r)).collect();
        let parent_id = |route: &Route| match route.parent {
            Some(index) => tree.routes[index].full_id.clone(),
            None => ROOT_ID.to_string(),
        };

        // (full_id, expected path, expected parent full id)
        let cases: &[(&str, Option<&str>, &str)] = &[
            ("/", Some("/"), ROOT_ID),
            ("/users", Some("/users"), ROOT_ID),
            ("/users/$userId", Some("/$userId"), "/users"),
            ("/users/", Some("/"), "/users"),
            ("/api/users", Some("/api/users"), ROOT_ID),
            ("/api/users/$userId", Some("/$userId"), "/api/users"),
            ("/_pathlessLayout", None, ROOT_ID),
            ("/_pathlessLayout/_nested-layout", None, "/_pathlessLayout"),
            (
                "/_pathlessLayout/_nested-layout/route-a",
                Some("/route-a"),
                "/_pathlessLayout/_nested-layout",
            ),
            // Trailing-`_` opt-out: parent is root, path strips the `_`, id keeps it.
            (
                "/posts_/$postId/deep",
                Some("/posts/$postId/deep"),
                ROOT_ID,
            ),
            // Escaped literal dot.
            ("/customScript.js", Some("/customScript.js"), ROOT_ID),
        ];
        for (id, expected_path, expected_parent) in cases {
            let route = by_id
                .get(id)
                .unwrap_or_else(|| panic!("missing generated route {id}"));
            assert_eq!(
                route.path.as_deref(),
                *expected_path,
                "path mismatch for {id}"
            );
            assert_eq!(parent_id(route), *expected_parent, "parent mismatch for {id}");
        }
    }

    #[test]
    fn emitted_source_is_runtime_complete() {
        let tree = generate_from_routes_dir(&fixture_routes_dir()).unwrap();
        let source = tree.emit();
        assert!(source.contains("import { Route as rootRouteImport } from './routes/__root'"));
        assert!(source.contains("export const routeTree = rootRouteImport._addFileChildren(rootRouteChildren)"));
        // Every route contributes a `.update(` const and an import.
        for route in &tree.routes {
            assert!(
                source.contains(&format!("const {}Route = {}RouteImport.update({{", route.var_base, route.var_base)),
                "missing update for {}",
                route.full_id
            );
            assert!(
                source.contains(&format!("from '{}'", route.import_path)),
                "missing import for {}",
                route.full_id
            );
        }
        // A route with children is referenced through its WithChildren variant.
        assert!(source.contains("PostsRouteWithChildren"));
    }

    #[test]
    fn unclassifiable_filename_is_a_hard_error() {
        // A doubled separator yields an empty segment: unclassifiable.
        let error = classify_filename("posts..deep").unwrap_err();
        assert!(error.contains("empty path segment"), "got: {error}");

        // Unbalanced escape brackets are rejected too.
        assert!(classify_filename("weird[.").is_err());
        assert!(classify_filename("weird.]js").is_err());
    }

    #[test]
    fn generate_hard_errors_and_names_the_bad_file() {
        let dir = tempfile::tempdir().unwrap();
        let routes = dir.path().join("routes");
        fs::create_dir_all(&routes).unwrap();
        fs::write(routes.join("__root.tsx"), "export const Route = {}\n").unwrap();
        // `foo..bar.tsx` -> stem `foo..bar` -> empty middle segment.
        fs::write(routes.join("foo..bar.tsx"), "export const Route = {}\n").unwrap();
        let error = generate_from_routes_dir(&routes).unwrap_err();
        assert!(error.contains("foo..bar"), "error should name the file: {error}");
        assert!(error.contains("cannot classify"), "got: {error}");
    }

    #[test]
    fn missing_root_route_is_a_hard_error() {
        let dir = tempfile::tempdir().unwrap();
        let routes = dir.path().join("routes");
        fs::create_dir_all(&routes).unwrap();
        fs::write(routes.join("index.tsx"), "export const Route = {}\n").unwrap();
        let error = generate_from_routes_dir(&routes).unwrap_err();
        assert!(error.contains("no root route"), "got: {error}");
    }

    #[test]
    fn route_ignore_parses_literal_alternation_and_rejects_regex() {
        // The literal-alternation forms real apps use, wrapped or bare.
        assert!(RouteIgnore::parse(Some("(_components|api)"))
            .unwrap()
            .excludes("_components", "_components/ui/x"));
        assert!(RouteIgnore::parse(Some("_components|api"))
            .unwrap()
            .excludes("x", "app/api/handler"));
        assert!(!RouteIgnore::parse(Some("(_components|api)"))
            .unwrap()
            .excludes("index", "index"));
        // The `-`/`.` colocation convention always applies, even with no pattern.
        assert!(RouteIgnore::none().excludes("-components", "settings/-components"));
        assert!(RouteIgnore::none().excludes(".DS_Store", ".DS_Store"));
        assert!(!RouteIgnore::none().excludes("index", "index"));
        // A pattern with real regex metacharacters is a hard, specific error, not a
        // silent mis-match.
        let error = RouteIgnore::parse(Some(r"\.stories\.")).unwrap_err();
        assert!(error.contains("regex features"), "got: {error}");
    }

    #[test]
    fn custom_routes_directory_drives_the_import_prefix_and_ignore() {
        // oc-web's shape: routesDirectory "app", an ignore pattern, colocated assets
        // and a `-` colocation dir. Imports must point at `./app/...` (including the
        // root), the ignored/colocated files must not become routes, and a
        // non-route asset in the routes dir must be skipped, not error.
        let dir = tempfile::tempdir().unwrap();
        let app = dir.path().join("app");
        fs::create_dir_all(app.join("settings/-components")).unwrap();
        fs::create_dir_all(app.join("_components/ui")).unwrap();
        fs::create_dir_all(app.join("api")).unwrap();
        fs::write(app.join("__root.tsx"), "export const Route = {}\n").unwrap();
        fs::write(app.join("index.tsx"), "export const Route = {}\n").unwrap();
        fs::write(app.join("settings/configuration.tsx"), "export const Route = {}\n").unwrap();
        fs::write(app.join("favicon.ico"), "not a route").unwrap();
        fs::write(app.join("settings/-components/Panel.tsx"), "export default 1\n").unwrap();
        fs::write(app.join("_components/ui/button.tsx"), "export default 1\n").unwrap();
        fs::write(app.join("api/health.tsx"), "export default 1\n").unwrap();

        let ignore = RouteIgnore::parse(Some("(_components|api)")).unwrap();
        let tree = generate_from_routes_dir_with(&app, "app", &ignore).unwrap();

        // Exactly the two real routes (root is separate), nothing from the ignored
        // or colocated dirs, no error on favicon.ico.
        let ids: Vec<&str> = tree.routes.iter().map(|route| route.full_id.as_str()).collect();
        assert_eq!(tree.routes.len(), 2, "got routes: {ids:?}");
        assert!(ids.contains(&"/"), "{ids:?}");
        assert!(ids.iter().any(|id| id.contains("settings/configuration")), "{ids:?}");

        // Imports resolve under `./app/...`, including the root.
        let emitted = tree.emit();
        assert!(emitted.contains("from './app/__root'"), "{emitted}");
        assert!(emitted.contains("from './app/index'"), "{emitted}");
        assert!(emitted.contains("from './app/settings/configuration'"), "{emitted}");
        assert!(!emitted.contains("./routes/"), "no hardcoded routes dir: {emitted}");
        assert!(!emitted.contains("_components") && !emitted.contains("/api/") && !emitted.contains("-components"), "{emitted}");
    }
}
