use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::fs;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use oxc_resolver::{ResolveOptions, Resolver, SideEffects, TsconfigDiscovery};
use oxc_sourcemap::SourceMapBuilder;
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::frontend_profile::{self, Phase};
use crate::resource_id::{LoaderKind, ResourceId};
use crate::transform::{DependencyDemand, FlatModule, FoldExpression, transform_module};

pub type ModuleId = String;
type DenseModuleId = usize;
type SharedModuleId = Arc<str>;

#[derive(Debug, Clone)]
struct ModuleState {
    hash: u64,
    dependencies: Vec<(String, DenseModuleId, DependencyDemand)>,
    pruned_imports: HashSet<String>,
    source: SharedModuleId,
    flat_module: Option<FlatModule>,
    code: String,
    assets: Vec<AssetEmit>,
    /// Stylesheet text contributed by a global CSS side-effect import
    /// (`import "./app.css"`). Extracted and concatenated into the output
    /// stylesheet in module execution order; `None` for a normal JS module.
    css: Option<String>,
    /// External specifiers (Node built-ins) this module imports. Left in the
    /// output for the runtime to resolve; a module with externals renders through
    /// the runtime path, since the flat path cannot bind an external.
    externals: Vec<String>,
}

struct LoadedModule {
    hash: u64,
    dependencies: Vec<(String, SharedModuleId, DependencyDemand)>,
    pruned_imports: HashSet<String>,
    source: SharedModuleId,
    flat_module: Option<FlatModule>,
    code: String,
    diagnostics: Vec<String>,
    assets: Vec<AssetEmit>,
    css: Option<String>,
    externals: Vec<String>,
}

/// A static asset (e.g. a `?url` import target) that must be content-hashed and
/// copied into the output `assets/` directory. The synthetic JavaScript module
/// that references it exports the public URL `/assets/<public_name>`.
#[derive(Debug, Clone)]
struct AssetEmit {
    source: PathBuf,
    public_name: String,
}

struct ResolutionCache {
    directories: [Mutex<HashMap<PathBuf, Arc<DirectoryResolutionCache>>>; 16],
    /// Plugin-host aliases: `(specifier, absolute_target)` applied as an exact-
    /// match rewrite before the standards-aware resolver runs. Shared read-only,
    /// so cheap to clone into each directory cache.
    aliases: Arc<Vec<(String, PathBuf)>>,
}

struct DirectoryResolutionCache {
    directory: PathBuf,
    specifiers: [Mutex<HashMap<String, Result<ResolvedModule, String>>>; 64],
    aliases: Arc<Vec<(String, PathBuf)>>,
}

#[derive(Clone)]
struct ResolvedModule {
    id: SharedModuleId,
    side_effect_free: bool,
}

impl ResolutionCache {
    fn new(aliases: Vec<(String, PathBuf)>) -> Self {
        Self {
            directories: std::array::from_fn(|_| Mutex::new(HashMap::new())),
            aliases: Arc::new(aliases),
        }
    }

    fn directory(&self, importer: &Path) -> Arc<DirectoryResolutionCache> {
        let importer_directory = importer.parent().unwrap_or_else(|| Path::new("."));
        let hash = hash_value(importer_directory);
        let mut shard = self.directories[hash as usize % self.directories.len()]
            .lock()
            .expect("resolution directory cache poisoned");
        if let Some(cache) = shard.get(importer_directory) {
            return Arc::clone(cache);
        }
        let cache = Arc::new(DirectoryResolutionCache {
            directory: importer_directory.to_path_buf(),
            specifiers: std::array::from_fn(|_| Mutex::new(HashMap::new())),
            aliases: Arc::clone(&self.aliases),
        });
        shard.insert(importer_directory.to_path_buf(), Arc::clone(&cache));
        cache
    }
}

impl DirectoryResolutionCache {
    fn resolve(
        &self,
        resolver: &Resolver,
        importer: &Path,
        specifier: &str,
    ) -> Result<ResolvedModule, String> {
        let hash = hash_value(specifier);
        let shard = &self.specifiers[hash as usize % self.specifiers.len()];
        if let Some(result) = shard
            .lock()
            .expect("resolution specifier cache poisoned")
            .get(specifier)
            .cloned()
        {
            return result;
        }
        // A specifier may carry a loader query and/or fragment (`app.css?url`,
        // `route.tsx?tsr-split=component`). Only the path component is a
        // filesystem concern; the query is re-attached to the resolved id and
        // interpreted later, at load time. A query never causes a resolve error.
        let resource = ResourceId::parse(specifier);
        let path_specifier = resource.path.as_str();
        // Plugin-host aliases win as an exact-match rewrite before the
        // standards-aware resolver (which would route a `#`-specifier through
        // package `imports` and fail). The alias target is a real file.
        if let Some((_, target)) = self
            .aliases
            .iter()
            .find(|(from, _)| from == path_specifier)
        {
            let result = if target.is_file() {
                Ok(ResolvedModule {
                    id: module_id_with_resource(target, &resource),
                    side_effect_free: false,
                })
            } else {
                Err(format!(
                    "alias {path_specifier:?} points to {}, which is not a file",
                    target.display()
                ))
            };
            shard
                .lock()
                .expect("resolution specifier cache poisoned")
                .insert(specifier.to_owned(), result.clone());
            return result;
        }
        // Most module graphs overwhelmingly use explicit relative files. Avoid
        // the general Node resolver on a cache miss when that exact file exists;
        // all ambiguous cases still take the standards-aware path.
        let exact_relative = path_specifier.strip_prefix("./").and_then(|relative| {
            let candidate = self.directory.join(relative);
            candidate
                .is_file()
                .then(|| module_id_with_resource(&candidate, &resource))
        });
        let result = if let Some(resolved) = exact_relative {
            Ok(ResolvedModule {
                id: resolved,
                side_effect_free: false,
            })
        } else {
            resolver
                .resolve_file(importer, path_specifier)
                .map_err(|error| error.to_string())
                .and_then(|resolution| {
                    let resolved = resolution.full_path();
                    if resolved.extension().and_then(|value| value.to_str()) == Some("node") {
                        Err(format!("native module {specifier:?} is not supported"))
                    } else {
                        let side_effect_free = resolution.package_json().is_some_and(|package| {
                            matches!(package.side_effects(), Some(SideEffects::Bool(false)))
                        });
                        Ok(ResolvedModule {
                            id: module_id_with_resource(&resolved, &resource),
                            side_effect_free,
                        })
                    }
                })
        };
        shard
            .lock()
            .expect("resolution specifier cache poisoned")
            .insert(specifier.to_owned(), result.clone());
        result
    }
}

fn hash_value(value: impl Hash) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[derive(Debug)]
pub struct BuildUpdate {
    pub delta: GraphDelta,
    pub transformed_modules: usize,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct GraphDelta {
    pub edge_updates: Vec<((ModuleId, ModuleId), isize)>,
    pub changed: BTreeSet<ModuleId>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EmitOptions {
    pub source_map: bool,
    pub minify: bool,
}

/// A count of what a client `public/` build wrote to disk: JavaScript chunks,
/// extracted stylesheets, and content-hashed assets. Counted from the emitted
/// files, not predicted, so the summary always matches reality.
#[derive(Debug, Clone)]
pub struct PublicBuildSummary {
    pub public_dir: PathBuf,
    pub javascript_files: usize,
    pub css_files: usize,
    pub asset_files: usize,
}

impl PublicBuildSummary {
    /// Walks an emitted `public/` directory and classifies each file: anything
    /// under `assets/` is a content-hashed asset; otherwise `.js`/`.css` by
    /// extension. Files with any other extension are ignored (there are none
    /// today, but the count stays honest if that changes).
    fn of(public_dir: &Path) -> Result<Self, String> {
        let mut summary = Self {
            public_dir: public_dir.to_path_buf(),
            javascript_files: 0,
            css_files: 0,
            asset_files: 0,
        };
        let mut stack = vec![public_dir.to_path_buf()];
        while let Some(directory) = stack.pop() {
            let entries = fs::read_dir(&directory)
                .map_err(|error| format!("cannot read {}: {error}", directory.display()))?;
            for entry in entries {
                let entry =
                    entry.map_err(|error| format!("cannot read {}: {error}", directory.display()))?;
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                let under_assets = path
                    .parent()
                    .and_then(|parent| parent.file_name())
                    .and_then(|name| name.to_str())
                    == Some("assets");
                if under_assets {
                    summary.asset_files += 1;
                } else {
                    match path.extension().and_then(|value| value.to_str()) {
                        Some("js") => summary.javascript_files += 1,
                        Some("css") => summary.css_files += 1,
                        _ => {}
                    }
                }
            }
        }
        Ok(summary)
    }
}

#[derive(Debug, Clone)]
pub struct VisualizationGraph {
    pub entry: String,
    pub nodes: Vec<VisualizationNode>,
    pub edges: Vec<VisualizationEdge>,
}

#[derive(Debug, Clone)]
pub struct VisualizationNode {
    pub id: String,
    pub dense_id: usize,
    pub reachable: bool,
    pub is_entry: bool,
    pub source_bytes: usize,
    pub lowered_bytes: usize,
    pub flat_eligible: bool,
    pub has_direct_effects: bool,
    pub declarations: Vec<String>,
    pub exports: Vec<String>,
    pub foldable_constants: Vec<String>,
    pub foldable_effects: Vec<String>,
    pub pruned_imports: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct VisualizationEdge {
    pub source: usize,
    pub target: usize,
    pub specifier: String,
    pub dynamic: bool,
    pub all: bool,
    pub names: Vec<String>,
}

#[derive(Debug, Default)]
pub struct DirectReachabilityUpdate {
    pub added: BTreeSet<ModuleId>,
    pub removed: BTreeSet<ModuleId>,
    pub used_full_recompute: bool,
}

/// A compact, persistent single-entry reachability index.
///
/// The selected parent edges form a spanning tree. Removing a non-tree edge
/// cannot affect reachability. Removing a tree edge repairs only its detached
/// subtree, unless that subtree is large enough that a dense full traversal is
/// cheaper.
pub struct DirectReachability {
    ids: Vec<SharedModuleId>,
    indices: HashMap<SharedModuleId, usize>,
    outgoing: Vec<Vec<usize>>,
    incoming: Vec<Vec<usize>>,
    reachable: Vec<bool>,
    parent: Vec<Option<usize>>,
    tree_children: Vec<Vec<usize>>,
    subtree_marks: Vec<u32>,
    mark_epoch: u32,
    entry: usize,
    reachable_count: usize,
}

pub struct Bundler {
    entry: DenseModuleId,
    ids: Vec<SharedModuleId>,
    indices: HashMap<SharedModuleId, DenseModuleId>,
    resolver: Resolver,
    resolution_cache: ResolutionCache,
    frontend_pool: ThreadPool,
    modules: Vec<Option<ModuleState>>,
}

impl Bundler {
    pub fn discover(entry: &Path) -> Result<(Self, BuildUpdate), String> {
        Self::discover_inner(entry, &BuildConfig::default())
    }

    pub fn discover_direct(entry: &Path) -> Result<(Self, BuildUpdate), String> {
        Self::discover_inner(entry, &BuildConfig::default())
    }

    /// Like [`Self::discover_direct`] but with build configuration, currently the
    /// resolver aliases a plugin host supplies (e.g. TanStack's
    /// `#tanstack-router-entry` -> the app's router). Aliases are baked into the
    /// resolver once, so incremental rebuilds pay no per-edit cost for them.
    pub fn discover_direct_with_config(
        entry: &Path,
        config: &BuildConfig,
    ) -> Result<(Self, BuildUpdate), String> {
        Self::discover_inner(entry, config)
    }

    fn discover_inner(entry: &Path, config: &BuildConfig) -> Result<(Self, BuildUpdate), String> {
        let entry_path = entry
            .canonicalize()
            .map_err(|error| format!("cannot open entry {}: {error}", entry.display()))?;
        let entry_id = module_id(&entry_path);
        let resolver = Resolver::new(resolve_options(config));
        let frontend_threads = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(4);
        let mut bundler = Self {
            entry: 0,
            ids: Vec::new(),
            indices: HashMap::new(),
            resolver,
            resolution_cache: ResolutionCache::new(
                config
                    .aliases
                    .iter()
                    .map(|(from, to)| (from.clone(), PathBuf::from(to)))
                    .collect(),
            ),
            frontend_pool: ThreadPoolBuilder::new()
                .num_threads(frontend_threads)
                .thread_name(|index| format!("diffpack-frontend-{index}"))
                .build()
                .map_err(|error| format!("cannot create frontend worker pool: {error}"))?,
            modules: Vec::new(),
        };
        bundler.entry = bundler.intern(entry_id.clone());

        let mut delta = GraphDelta::default();
        let mut diagnostics = Vec::new();
        let transformed_modules =
            bundler.discover_from(vec![entry_id], &mut delta, &mut diagnostics, false)?;
        Ok((
            bundler,
            BuildUpdate {
                delta,
                transformed_modules,
                diagnostics,
            },
        ))
    }

    pub fn rebuild_path(&mut self, path: &Path) -> Result<BuildUpdate, String> {
        let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let id = module_id(&path);
        let Some(&index) = self.indices.get(id.as_ref()) else {
            return Ok(BuildUpdate {
                delta: GraphDelta::default(),
                transformed_modules: 0,
                diagnostics: Vec::new(),
            });
        };
        let Some(old) = self.modules[index].clone() else {
            return Ok(BuildUpdate {
                delta: GraphDelta::default(),
                transformed_modules: 0,
                diagnostics: Vec::new(),
            });
        };

        let mut delta = GraphDelta::default();
        let mut diagnostics = Vec::new();
        if !path.is_file() {
            delta.changed.insert(id.to_string());
            for (_, target, _) in &old.dependencies {
                delta
                    .edge_updates
                    .push(((id.to_string(), self.ids[*target].to_string()), -1));
            }
            self.modules[index] = None;
            return Ok(BuildUpdate {
                delta,
                transformed_modules: 0,
                diagnostics,
            });
        }

        let new = self.load_module(&path, &mut diagnostics)?;
        if old.hash != new.hash {
            delta.changed.insert(id.to_string());
        }
        let old_edges = old
            .dependencies
            .iter()
            .map(|(_, target, _)| target)
            .map(|target| (id.to_string(), self.ids[*target].to_string()))
            .collect::<BTreeSet<_>>();
        let new_edges = new
            .dependencies
            .iter()
            .map(|(_, target, _)| target)
            .map(|target| (id.to_string(), self.ids[*target].to_string()))
            .collect::<BTreeSet<_>>();
        delta.edge_updates.extend(
            old_edges
                .difference(&new_edges)
                .cloned()
                .map(|edge| (edge, -1)),
        );
        delta.edge_updates.extend(
            new_edges
                .difference(&old_edges)
                .cloned()
                .map(|edge| (edge, 1)),
        );
        let new_paths = new
            .dependencies
            .iter()
            .map(|(_, target, _)| target)
            .filter(|dependency| self.modules[**dependency].is_none())
            .map(|dependency| self.ids[*dependency].clone())
            .collect::<Vec<_>>();
        self.modules[index] = Some(new);
        let transformed_modules =
            1 + self.discover_from(new_paths, &mut delta, &mut diagnostics, true)?;

        Ok(BuildUpdate {
            delta,
            transformed_modules,
            diagnostics,
        })
    }

    pub fn emit(&self, reachable: &BTreeSet<ModuleId>, output: &Path) -> Result<(), String> {
        self.emit_with_options(reachable, output, EmitOptions::default())
    }

    /// Emits the client browser build into `<output_root>/public/`: the entry
    /// JavaScript chunk (`client.js`), its dynamic-import chunks, the extracted
    /// stylesheet, and every content-hashed asset under `public/assets/`. The
    /// `public/` directory is rebuilt from scratch so stale files never linger,
    /// and the returned [`PublicBuildSummary`] counts exactly what landed on disk.
    ///
    /// This drives the existing single-output emit at a `public/` layout; it is a
    /// build-time entry point (off the incremental hot path), so the thesis guards
    /// are unaffected.
    pub fn emit_public(
        &self,
        reachable: &BTreeSet<ModuleId>,
        output_root: &Path,
        options: EmitOptions,
    ) -> Result<PublicBuildSummary, String> {
        let public_dir = output_root.join("public");
        if public_dir.exists() {
            fs::remove_dir_all(&public_dir).map_err(|error| {
                format!("cannot clear {}: {error}", public_dir.display())
            })?;
        }
        let entry_output = public_dir.join("client.js");
        self.emit_with_options(reachable, &entry_output, options)?;
        PublicBuildSummary::of(&public_dir)
    }

    pub fn emit_with_options(
        &self,
        reachable: &BTreeSet<ModuleId>,
        output: &Path,
        options: EmitOptions,
    ) -> Result<(), String> {
        let parent = output
            .parent()
            .ok_or_else(|| format!("output has no parent: {}", output.display()))?;
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
        let reachable_dense = reachable
            .iter()
            .filter_map(|id| self.indices.get(id.as_str()).copied())
            .collect::<Vec<_>>();
        let allowed = reachable_dense.iter().copied().collect::<HashSet<_>>();
        self.emit_assets(&allowed, parent)?;
        self.emit_css(&allowed, output)?;
        let mut runtime_ids = vec![None; self.ids.len()];
        for (runtime_id, dense_id) in reachable_dense.into_iter().enumerate() {
            runtime_ids[dense_id] = Some(runtime_id);
        }
        let main_modules = self.static_closure(self.entry, &allowed);
        let main_set = main_modules.iter().copied().collect::<HashSet<_>>();
        let mut dynamic_roots = allowed
            .iter()
            .flat_map(|source| {
                self.modules[*source]
                    .iter()
                    .flat_map(|module| module.dependencies.iter())
                    .filter(|(_, _, demand)| demand.dynamic)
                    .map(|(_, target, _)| *target)
            })
            .filter(|target| !main_set.contains(target))
            .collect::<Vec<_>>();
        dynamic_roots.sort_by(|left, right| self.ids[*left].cmp(&self.ids[*right]));
        dynamic_roots.dedup();
        let mut chunk_names = HashMap::with_capacity(dynamic_roots.len());
        for (index, root) in dynamic_roots.iter().copied().enumerate() {
            let chunk_path = chunk_path(output, index + 1)?;
            let chunk_file = chunk_path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| format!("chunk path is not UTF-8: {}", chunk_path.display()))?;
            chunk_names.insert(root, format!("./{chunk_file}"));
        }
        for (index, root) in dynamic_roots.iter().copied().enumerate() {
            let modules = self.static_closure(root, &allowed);
            let rendered = self.render_best(&modules, root, &chunk_names, &runtime_ids, false);
            self.write_rendered(rendered, &chunk_path(output, index + 1)?, options)?;
        }
        if options.minify
            && !options.source_map
            && dynamic_roots.is_empty()
            && let Some(rendered) = self.render_folded_constants(&main_modules)
        {
            return self.write_rendered(rendered, output, options);
        }
        let rendered =
            self.render_best(&main_modules, self.entry, &chunk_names, &runtime_ids, true);
        self.write_rendered(rendered, output, options)
    }

    /// Copies every content-hashed asset referenced by a reachable module into
    /// `<output_dir>/assets/`. Deduplicated by public name, so an asset imported
    /// from several modules is written once.
    fn emit_assets(&self, allowed: &HashSet<DenseModuleId>, parent: &Path) -> Result<(), String> {
        let mut written = HashSet::new();
        let mut assets_dir_ready = false;
        for &dense in allowed {
            let Some(module) = self.modules[dense].as_ref() else {
                continue;
            };
            for asset in &module.assets {
                if !written.insert(asset.public_name.clone()) {
                    continue;
                }
                let assets_dir = parent.join("assets");
                if !assets_dir_ready {
                    fs::create_dir_all(&assets_dir).map_err(|error| {
                        format!("cannot create {}: {error}", assets_dir.display())
                    })?;
                    assets_dir_ready = true;
                }
                let destination = assets_dir.join(&asset.public_name);
                fs::copy(&asset.source, &destination).map_err(|error| {
                    format!(
                        "cannot copy asset {} to {}: {error}",
                        asset.source.display(),
                        destination.display()
                    )
                })?;
            }
        }
        Ok(())
    }

    /// Extracts the stylesheet: concatenates every reachable global CSS module's
    /// text in module execution order and writes it beside the bundle as
    /// `<output_stem>.css`. Nothing is written when no CSS is imported.
    fn emit_css(&self, allowed: &HashSet<DenseModuleId>, output: &Path) -> Result<(), String> {
        let order = self.static_execution_order(allowed).unwrap_or_else(|| {
            let mut ids = allowed.iter().copied().collect::<Vec<_>>();
            ids.sort_by(|left, right| self.ids[*left].cmp(&self.ids[*right]));
            ids
        });
        let mut stylesheet = String::new();
        for dense in order {
            if let Some(module) = self.modules[dense].as_ref()
                && let Some(css) = &module.css
            {
                if !stylesheet.is_empty() {
                    stylesheet.push('\n');
                }
                stylesheet.push_str(css);
            }
        }
        if stylesheet.is_empty() {
            return Ok(());
        }
        let css_path = output.with_extension("css");
        fs::write(&css_path, stylesheet)
            .map_err(|error| format!("cannot write {}: {error}", css_path.display()))
    }

    fn render_folded_constants(&self, modules: &[DenseModuleId]) -> Option<RenderedBundle> {
        let included = modules.iter().copied().collect::<HashSet<_>>();
        let order = self.static_execution_order(&included)?;
        let mut values = HashMap::<String, f64>::new();
        let mut output = String::new();
        for dense_index in order {
            let foldable = self.modules[dense_index]
                .as_ref()?
                .flat_module
                .as_ref()?
                .foldable
                .as_ref()?;
            for (name, expression) in &foldable.constants {
                values.insert(name.clone(), evaluate_fold_expression(expression, &values)?);
            }
            for expression in &foldable.console_logs {
                let value = evaluate_fold_expression(expression, &values)?;
                output.push_str("console.log(");
                output.push_str(&format_javascript_number(value)?);
                output.push_str(");");
            }
        }
        Some(RenderedBundle {
            code: output,
            mappings: Vec::new(),
        })
    }

    fn write_rendered(
        &self,
        rendered: RenderedBundle,
        output: &Path,
        options: EmitOptions,
    ) -> Result<(), String> {
        let mut code = rendered.code;
        if options.source_map {
            let map_path = path_with_suffix(output, ".map");
            let map_name = map_path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| format!("source-map path is not UTF-8: {}", map_path.display()))?;
            let output_name = output
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| format!("output path is not UTF-8: {}", output.display()))?;
            fs::write(&map_path, self.source_map(&rendered.mappings, output_name))
                .map_err(|error| format!("cannot write {}: {error}", map_path.display()))?;
            code.push_str(&format!("\n//# sourceMappingURL={map_name}\n"));
        }
        fs::write(output, code)
            .map_err(|error| format!("cannot write {}: {error}", output.display()))
    }

    fn static_closure(
        &self,
        root: DenseModuleId,
        allowed: &HashSet<DenseModuleId>,
    ) -> Vec<DenseModuleId> {
        let mut seen = HashSet::new();
        let mut pending = vec![root];
        while let Some(source) = pending.pop() {
            if !allowed.contains(&source) || !seen.insert(source) {
                continue;
            }
            if let Some(module) = &self.modules[source] {
                pending.extend(
                    module
                        .dependencies
                        .iter()
                        .filter(|(_, _, demand)| !demand.dynamic)
                        .map(|(_, target, _)| *target),
                );
            }
        }
        let mut modules = seen.into_iter().collect::<Vec<_>>();
        modules.sort_by(|left, right| self.ids[*left].cmp(&self.ids[*right]));
        modules
    }

    pub fn all_modules(&self) -> BTreeSet<ModuleId> {
        self.modules
            .iter()
            .enumerate()
            .filter(|(_, module)| module.is_some())
            .map(|(index, _)| self.ids[index].to_string())
            .collect()
    }

    /// Builds a persistent dense reachability index for incremental edits.
    pub fn direct_reachability(&self) -> DirectReachability {
        DirectReachability::new(self)
    }

    /// Recomputes entry reachability from scratch using dense integer IDs.
    pub fn reachable_modules_direct(&self) -> BTreeSet<ModuleId> {
        self.direct_reachability().reachable_modules()
    }

    pub fn visualization_graph(&self, reachable: &BTreeSet<ModuleId>) -> VisualizationGraph {
        let nodes = self
            .modules
            .iter()
            .enumerate()
            .filter_map(|(dense_id, module)| {
                let module = module.as_ref()?;
                let flat = module.flat_module.as_ref();
                let foldable = flat.and_then(|flat| flat.foldable.as_ref());
                let mut pruned_imports = module.pruned_imports.iter().cloned().collect::<Vec<_>>();
                pruned_imports.sort();
                Some(VisualizationNode {
                    id: self.ids[dense_id].to_string(),
                    dense_id,
                    reachable: reachable.contains(self.ids[dense_id].as_ref()),
                    is_entry: dense_id == self.entry,
                    source_bytes: module.source.len(),
                    lowered_bytes: module.code.len(),
                    flat_eligible: flat.is_some(),
                    has_direct_effects: flat.is_none_or(|flat| flat.has_direct_effects),
                    declarations: flat.map_or_else(Vec::new, |flat| flat.declarations.clone()),
                    exports: flat.map_or_else(Vec::new, |flat| flat.exports.clone()),
                    foldable_constants: foldable.map_or_else(Vec::new, |foldable| {
                        foldable
                            .constants
                            .iter()
                            .map(|(name, expression)| {
                                format!("{name} = {}", display_fold_expression(expression))
                            })
                            .collect()
                    }),
                    foldable_effects: foldable.map_or_else(Vec::new, |foldable| {
                        foldable
                            .console_logs
                            .iter()
                            .map(|expression| {
                                format!("console.log({})", display_fold_expression(expression))
                            })
                            .collect()
                    }),
                    pruned_imports,
                })
            })
            .collect::<Vec<_>>();
        let edges = self
            .modules
            .iter()
            .enumerate()
            .flat_map(|(source, module)| {
                module.iter().flat_map(move |module| {
                    module
                        .dependencies
                        .iter()
                        .map(move |(specifier, target, demand)| VisualizationEdge {
                            source,
                            target: *target,
                            specifier: specifier.clone(),
                            dynamic: demand.dynamic,
                            all: demand.all,
                            names: demand.names.clone(),
                        })
                })
            })
            .collect();
        VisualizationGraph {
            entry: self.ids[self.entry].to_string(),
            nodes,
            edges,
        }
    }

    pub fn watch_root(&self) -> PathBuf {
        PathBuf::from(self.ids[self.entry].as_ref())
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf()
    }

    pub fn worker_count(&self) -> usize {
        self.frontend_pool.current_num_threads()
    }

    fn discover_from(
        &mut self,
        paths: Vec<SharedModuleId>,
        delta: &mut GraphDelta,
        diagnostics: &mut Vec<String>,
        record_delta: bool,
    ) -> Result<usize, String> {
        let mut frontier = paths.into_iter().collect::<BTreeSet<_>>();
        let mut transformed = 0;
        while !frontier.is_empty() {
            let paths = frontier
                .into_iter()
                .filter(|path| {
                    self.indices
                        .get(path.as_ref())
                        .is_none_or(|index| self.modules[*index].is_none())
                })
                .collect::<Vec<_>>();
            let loaded = self.frontend_pool.install(|| {
                paths
                    .into_par_iter()
                    .map(|path| {
                        let result = load_uncached(
                            &self.resolver,
                            &self.resolution_cache,
                            Path::new(path.as_ref()),
                        );
                        (path, result)
                    })
                    .collect::<Vec<_>>()
            });
            frontier = BTreeSet::new();
            for (path, result) in loaded {
                if self
                    .indices
                    .get(path.as_ref())
                    .is_some_and(|index| self.modules[*index].is_some())
                {
                    continue;
                }
                let loaded = result?;
                diagnostics.extend(loaded.diagnostics);
                transformed += 1;
                let source = self.intern(path.clone());
                let mut dependencies = Vec::with_capacity(loaded.dependencies.len());
                for (specifier, target, demand) in loaded.dependencies {
                    let target_index = self.intern(target.clone());
                    if record_delta {
                        delta
                            .edge_updates
                            .push(((path.to_string(), target.to_string()), 1));
                    }
                    if self.modules[target_index].is_none() {
                        frontier.insert(target.clone());
                    }
                    dependencies.push((specifier, target_index, demand));
                }
                self.modules[source] = Some(ModuleState {
                    hash: loaded.hash,
                    dependencies,
                    pruned_imports: loaded.pruned_imports,
                    source: loaded.source,
                    flat_module: loaded.flat_module,
                    code: loaded.code,
                    assets: loaded.assets,
                    css: loaded.css,
                    externals: loaded.externals,
                });
            }
        }
        Ok(transformed)
    }

    fn intern(&mut self, id: SharedModuleId) -> DenseModuleId {
        if let Some(&index) = self.indices.get(id.as_ref()) {
            return index;
        }
        let index = self.ids.len();
        self.ids.push(id.clone());
        self.indices.insert(id, index);
        self.modules.push(None);
        index
    }

    fn load_module(
        &mut self,
        path: &Path,
        diagnostics: &mut Vec<String>,
    ) -> Result<ModuleState, String> {
        let id = module_id(path);
        // A loader (query, stylesheet, or asset) may claim this id before it is
        // ever read as JavaScript.
        if let Some(special) = load_special_module(&id, path) {
            let special = special?;
            return Ok(ModuleState {
                hash: special.hash,
                dependencies: Vec::new(),
                pruned_imports: HashSet::new(),
                source: id.clone(),
                flat_module: special.flat_module,
                code: special.code,
                assets: special.assets,
                css: special.css,
                externals: Vec::new(),
            });
        }
        let read_started = frontend_profile::start();
        let source = fs::read_to_string(path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        frontend_profile::finish(Phase::Read, read_started);
        let hash = content_hash(source.as_bytes());
        if let Some(current) = self
            .indices
            .get(id.as_ref())
            .and_then(|index| self.modules[*index].as_ref())
            && current.hash == hash
        {
            return Ok(current.clone());
        }
        let transformed = transform_module(path, &source);
        diagnostics.extend(
            transformed
                .diagnostics
                .iter()
                .map(|diagnostic| format!("{}: {diagnostic}", path.display())),
        );

        let resolved_dependencies = resolve_dependencies(
            &self.resolver,
            &self.resolution_cache,
            path,
            &transformed.dependencies,
            &transformed.dependency_demands,
            diagnostics,
        );
        let dependencies = resolved_dependencies
            .dependencies
            .into_iter()
            .map(|(specifier, target, demand)| (specifier, self.intern(target), demand))
            .collect();

        Ok(ModuleState {
            hash,
            dependencies,
            pruned_imports: resolved_dependencies.pruned_imports,
            source: Arc::from(source),
            flat_module: transformed.flat_module,
            code: transformed.code,
            assets: Vec::new(),
            css: None,
            externals: resolved_dependencies.externals,
        })
    }

    fn render_best(
        &self,
        reachable: &[DenseModuleId],
        entry: DenseModuleId,
        chunk_names: &HashMap<DenseModuleId, String>,
        runtime_ids: &[Option<usize>],
        is_main: bool,
    ) -> RenderedBundle {
        self.render_flat(reachable, entry, chunk_names, is_main)
            .unwrap_or_else(|| {
                self.render_runtime(reachable, entry, chunk_names, runtime_ids, is_main)
            })
    }

    fn render_flat(
        &self,
        reachable: &[DenseModuleId],
        entry: DenseModuleId,
        chunk_names: &HashMap<DenseModuleId, String>,
        is_main: bool,
    ) -> Option<RenderedBundle> {
        let reachable_set = reachable.iter().copied().collect::<HashSet<_>>();
        for &dense_index in reachable {
            let module = self.modules[dense_index].as_ref()?;
            module.flat_module.as_ref()?;
            // The flat path strips import bindings and cannot bind an external's
            // `require`. A module with externals renders through the runtime path.
            if !module.externals.is_empty() {
                return None;
            }
        }
        let mut included = reachable
            .iter()
            .copied()
            .filter(|dense_index| {
                self.modules[*dense_index]
                    .as_ref()
                    .and_then(|module| module.flat_module.as_ref())
                    .is_some_and(|flat| flat.has_direct_effects)
            })
            .collect::<HashSet<_>>();
        if !is_main {
            included.insert(entry);
        }
        let mut pending = included.iter().copied().collect::<Vec<_>>();
        while let Some(source) = pending.pop() {
            let module = self.modules[source].as_ref()?;
            for (_, target, demand) in &module.dependencies {
                if !demand.dynamic
                    && reachable_set.contains(target)
                    && (demand.all || !demand.names.is_empty())
                    && included.insert(*target)
                {
                    pending.push(*target);
                }
            }
        }
        if included.is_empty() {
            return Some(RenderedBundle {
                code: String::new(),
                mappings: Vec::new(),
            });
        }
        let order = self.static_execution_order(&included)?;
        let mut declarations = HashSet::new();
        for &dense_index in &order {
            let flat = self.modules[dense_index].as_ref()?.flat_module.as_ref()?;
            if flat
                .declarations
                .iter()
                .any(|name| !declarations.insert(name.clone()))
            {
                return None;
            }
        }

        let mut demands = self.export_demands(reachable, entry);
        if is_main {
            demands[entry] = ExportDemand::default();
        }
        let mut code = String::new();
        let mut mappings = Vec::with_capacity(order.len());
        let mut generated_line = 0_u32;
        for dense_index in order {
            let module = self.modules[dense_index].as_ref()?;
            let flat = module.flat_module.as_ref()?;
            let mut module_code =
                shake_module_code(&flat.code, &demands[dense_index], &module.pruned_imports);
            for (specifier, target, demand) in &module.dependencies {
                if !demand.dynamic {
                    continue;
                }
                let chunk = chunk_names.get(target)?;
                let import = format!("import({})", quote(specifier));
                let lowered_import = format!("__dynamic(require, {})", quote(specifier));
                let replacement = format!("Promise.resolve().then(()=>require({}))", quote(chunk));
                if module_code.contains(&import) {
                    module_code = module_code.replace(&import, &replacement);
                } else if module_code.contains(&lowered_import) {
                    module_code = module_code.replace(&lowered_import, &replacement);
                } else {
                    return None;
                }
            }
            if !module_code.is_empty() {
                let generated_lines = module_code.lines().count() as u32;
                mappings.push(ModuleMapping {
                    dense_index,
                    generated_line,
                    generated_lines,
                });
                generated_line += generated_lines;
                code.push_str(&module_code);
            }
        }
        if !is_main {
            let exports = self.modules[entry]
                .as_ref()?
                .flat_module
                .as_ref()?
                .exports
                .iter()
                .filter(|name| demands[entry].includes(name))
                .cloned()
                .collect::<Vec<_>>();
            code.push_str(&format!("module.exports={{{}}};\n", exports.join(",")));
        }
        Some(RenderedBundle { code, mappings })
    }

    fn static_execution_order(
        &self,
        allowed: &HashSet<DenseModuleId>,
    ) -> Option<Vec<DenseModuleId>> {
        fn visit(
            bundler: &Bundler,
            source: DenseModuleId,
            allowed: &HashSet<DenseModuleId>,
            states: &mut HashMap<DenseModuleId, u8>,
            order: &mut Vec<DenseModuleId>,
        ) -> Option<()> {
            match states.get(&source) {
                Some(1) => return None,
                Some(2) => return Some(()),
                _ => {}
            }
            states.insert(source, 1);
            let module = bundler.modules[source].as_ref()?;
            for (_, target, demand) in &module.dependencies {
                if !demand.dynamic && allowed.contains(target) {
                    visit(bundler, *target, allowed, states, order)?;
                }
            }
            states.insert(source, 2);
            order.push(source);
            Some(())
        }

        let mut order = Vec::with_capacity(allowed.len());
        let mut roots = allowed.iter().copied().collect::<Vec<_>>();
        roots.sort_by(|left, right| self.ids[*left].cmp(&self.ids[*right]));
        let mut states = HashMap::new();
        for root in roots {
            visit(self, root, allowed, &mut states, &mut order)?;
        }
        (order.len() == allowed.len()).then_some(order)
    }

    fn render_runtime(
        &self,
        reachable: &[DenseModuleId],
        entry: DenseModuleId,
        chunk_names: &HashMap<DenseModuleId, String>,
        runtime_ids: &[Option<usize>],
        is_main: bool,
    ) -> RenderedBundle {
        let export_demands = self.export_demands(reachable, entry);
        let fragments = reachable
            .par_iter()
            .filter_map(|&dense_index| {
                let module = self.modules[dense_index].as_ref()?;
                let runtime_id = runtime_ids[dense_index]
                    .expect("a rendered module must have a deterministic runtime ID");
                let code = shake_module_code(
                    &module.code,
                    &export_demands[dense_index],
                    &module.pruned_imports,
                );
                let module_fragment = format!(
                    "{runtime_id}:function(module,exports,require,__toESM,__export,__reExport,__import,__dynamic,__esmNamespace,__seal){{\n{}\n}},\n",
                    code
                );
                let mut map_fragment = format!("{runtime_id}:{{");
                let mut chunk_fragment = format!("{runtime_id}:{{");
                for (specifier, target, demand) in &module.dependencies {
                    let target_runtime_id = runtime_ids[*target]
                        .expect("a reachable dependency must have a deterministic runtime ID");
                    map_fragment.push_str(&format!(
                        "{}:{target_runtime_id},",
                        quote(specifier)
                    ));
                    if demand.dynamic {
                        let chunk = chunk_names.get(target).map_or("null".to_owned(), |chunk| {
                            quote(chunk)
                        });
                        chunk_fragment.push_str(&format!(
                            "{}:[{chunk},{target_runtime_id}],",
                            quote(specifier)
                        ));
                    }
                }
                map_fragment.push_str("},\n");
                chunk_fragment.push_str("},\n");
                Some((
                    dense_index,
                    module_fragment,
                    map_fragment,
                    chunk_fragment,
                    code.lines().count(),
                ))
            })
            .collect::<Vec<_>>();
        let mut modules = String::new();
        let mut maps = String::new();
        let mut chunks = String::new();
        let mut mappings = Vec::with_capacity(fragments.len());
        let mut module_lines = 0_u32;
        for (dense_index, module, map, chunk, generated_lines) in fragments {
            mappings.push(ModuleMapping {
                dense_index,
                generated_line: 3 + module_lines,
                generated_lines: generated_lines as u32,
            });
            module_lines += module.matches('\n').count() as u32;
            modules.push_str(&module);
            maps.push_str(&map);
            chunks.push_str(&chunk);
        }

        let runtime_key = quote(&format!(
            "__diffpack_runtime:{}",
            self.ids[self.entry].as_ref()
        ));
        let entry_runtime_id =
            runtime_ids[entry].expect("a chunk entry must have a deterministic runtime ID");
        let tail = if is_main {
            format!(
                r#"const __runtime=globalThis[{runtime_key}]??=(()=>{{
const __modules=Object.create(null),__maps=Object.create(null),__chunks=Object.create(null),__cache=Object.create(null);
const __exportStates=new WeakMap();
function __esmNamespace(){{const namespace=Object.create(null);Object.defineProperty(namespace,Symbol.toStringTag,{{value:"Module"}});return namespace;}}
function __seal(namespace){{for(const key of Reflect.ownKeys(namespace)){{const descriptor=Object.getOwnPropertyDescriptor(namespace,key);if(descriptor?.configurable)Object.defineProperty(namespace,key,{{configurable:false}});}}Object.preventExtensions(namespace);}}
function __exportState(target){{let state=__exportStates.get(target);if(!state){{state={{explicit:new Set(),stars:new Map(),ambiguous:new Set()}};__exportStates.set(target,state);}}return state;}}
function __export(target,name,getter){{const state=__exportState(target);const descriptor=Object.getOwnPropertyDescriptor(target,name);if(descriptor?.configurable)delete target[name];if(!Object.prototype.hasOwnProperty.call(target,name))Object.defineProperty(target,name,{{enumerable:true,configurable:true,get:getter}});state.explicit.add(name);state.stars.delete(name);state.ambiguous.delete(name);}}
function __reExport(target,source){{const state=__exportState(target);for(const key of Object.keys(source)){{if(key==="default"||key==="__esModule"||state.explicit.has(key)||state.ambiguous.has(key))continue;const previous=state.stars.get(key);if(previous&&previous!==source){{delete target[key];state.stars.delete(key);state.ambiguous.add(key);continue;}}if(!previous){{Object.defineProperty(target,key,{{enumerable:true,configurable:true,get:()=>source[key]}});state.stars.set(key,source);}}}}}}
function __toESM(value){{
  if(value&&value.__esModule)return value;
  const namespace=Object.create(null);
  Object.defineProperty(namespace,"__esModule",{{value:true}});
  Object.defineProperty(namespace,"__diffpackCJS",{{value:true}});
  __export(namespace,"default",()=>value);
  if(value&&(typeof value==="object"||typeof value==="function"))for(const key of Object.keys(value))if(key!=="default")__export(namespace,key,()=>value[key]);
  return namespace;
}}
function __import(namespace,name){{if(Object.prototype.hasOwnProperty.call(namespace,name)||namespace.__diffpackCJS)return namespace[name];throw new SyntaxError("Module does not provide an export named "+name);}}
function __dynamic(require,specifier){{return Promise.resolve().then(()=>require.dynamic(specifier)).then(__toESM);}}
function __register(modules,maps,chunks){{Object.assign(__modules,modules);Object.assign(__maps,maps);Object.assign(__chunks,chunks);}}
function __require(id){{
  if(__cache[id])return __cache[id].exports;
  const factory=__modules[id];
  if(!factory)throw new Error("Module is not loaded: "+id);
  const module={{exports:{{}}}};
  __cache[id]=module;
  const require=specifier=>{{const target=__maps[id][specifier];if(target===undefined){{if(requireNative)return requireNative(specifier);throw new Error("Cannot resolve "+specifier+" from "+id);}}return __require(target);}};
  require.dynamic=specifier=>{{const chunk=__chunks[id][specifier];if(chunk===undefined)return require(specifier);if(chunk[0]!==null){{if(typeof requireNative!=="function")throw new Error("Dynamic chunks require a CommonJS host");return requireNative(chunk[0]);}}return __require(chunk[1]);}};
  factory(module,module.exports,require,__toESM,__export,__reExport,__import,__dynamic,__esmNamespace,__seal);
  return module.exports;
}}
const requireNative=typeof require==="function"?require:null;
return {{register:__register,require:__require}};
}})();
__runtime.register(__newModules,__newMaps,__newChunks);
return __runtime.require({entry_runtime_id});"#
            )
        } else {
            format!(
                r#"const __runtime=globalThis[{runtime_key}];
if(!__runtime)throw new Error("Diffpack runtime is not initialized");
__runtime.register(__newModules,__newMaps,__newChunks);
return __runtime.require({entry_runtime_id});"#
            )
        };
        let code = format!(
            r#"module.exports=(()=>{{
"use strict";
const __newModules={{{modules}}};
const __newMaps={{{maps}}};
const __newChunks={{{chunks}}};
{tail}
}})();
"#
        );
        RenderedBundle { code, mappings }
    }

    fn source_map(&self, mappings: &[ModuleMapping], output_name: &str) -> String {
        let mut builder = SourceMapBuilder::default();
        builder.set_file(output_name);
        for mapping in mappings {
            let module = self.modules[mapping.dense_index]
                .as_ref()
                .expect("rendered module must exist");
            let source_path = self.ids[mapping.dense_index].as_ref();
            let source_id = builder.add_source_and_content(source_path, module.source.as_ref());
            let source_lines = module.source.lines().count().max(1) as u32;
            for line in 0..mapping.generated_lines {
                builder.add_token(
                    mapping.generated_line + line,
                    0,
                    line.min(source_lines - 1),
                    0,
                    Some(source_id),
                    None,
                );
            }
        }
        builder.into_sourcemap().to_json_string()
    }

    fn export_demands(
        &self,
        reachable: &[DenseModuleId],
        entry: DenseModuleId,
    ) -> Vec<ExportDemand> {
        let mut demands = vec![ExportDemand::default(); self.modules.len()];
        demands[entry].all = true;
        for &source in reachable {
            let Some(module) = &self.modules[source] else {
                continue;
            };
            for (_, target, demand) in &module.dependencies {
                demands[*target].merge(ExportDemand {
                    all: demand.all,
                    names: demand.names.iter().cloned().collect(),
                });
            }
        }
        demands
    }
}

#[derive(Clone, Default)]
struct ExportDemand {
    all: bool,
    names: HashSet<String>,
}

impl ExportDemand {
    fn merge(&mut self, other: Self) {
        self.all |= other.all;
        self.names.extend(other.names);
    }

    fn includes(&self, name: &str) -> bool {
        self.all || self.names.contains(name)
    }
}

fn shake_module_code(
    code: &str,
    demand: &ExportDemand,
    pruned_imports: &HashSet<String>,
) -> String {
    let mut output = String::with_capacity(code.len());
    let mut skip_declaration = false;
    for line in code.lines() {
        if let Some(marked) = line.strip_prefix("/*__diffpack_import:")
            && let Some((specifier, code)) = marked.split_once("__*/")
            && let Ok(specifier) = serde_json::from_str::<String>(specifier)
        {
            if !pruned_imports.contains(&specifier) {
                output.push_str(code);
                output.push('\n');
            }
            continue;
        }
        if let Some(names) = line
            .strip_prefix("/*__diffpack_decl:")
            .and_then(|line| line.strip_suffix("__*/"))
        {
            skip_declaration = !names.split(',').any(|name| demand.includes(name));
            continue;
        }
        if line == "/*__diffpack_decl_end__*/" {
            skip_declaration = false;
            continue;
        }
        if skip_declaration {
            continue;
        }
        if let Some(marked) = line.strip_prefix("/*__diffpack_export:")
            && let Some((name, statement)) = marked.split_once("__*/")
        {
            if demand.includes(name) {
                output.push_str(statement);
                output.push('\n');
            }
            continue;
        }
        output.push_str(line);
        output.push('\n');
    }
    output
}

impl DirectReachability {
    const RECOMPUTE_NUMERATOR: usize = 1;
    const RECOMPUTE_DENOMINATOR: usize = 4;

    fn new(bundler: &Bundler) -> Self {
        let node_count = bundler.ids.len();
        let mut graph = Self {
            ids: bundler.ids.clone(),
            indices: bundler.indices.clone(),
            outgoing: vec![Vec::new(); node_count],
            incoming: vec![Vec::new(); node_count],
            reachable: vec![false; node_count],
            parent: vec![None; node_count],
            tree_children: vec![Vec::new(); node_count],
            subtree_marks: vec![0; node_count],
            mark_epoch: 0,
            entry: bundler.entry,
            reachable_count: 0,
        };

        for (source, module) in bundler.modules.iter().enumerate() {
            let Some(module) = module else {
                continue;
            };
            for (_, target, _) in &module.dependencies {
                graph.insert_edge(source, *target);
            }
        }
        graph.recompute();
        graph
    }

    pub fn reachable_modules(&self) -> BTreeSet<ModuleId> {
        self.reachable
            .iter()
            .enumerate()
            .filter(|(_, reachable)| **reachable)
            .map(|(index, _)| self.ids[index].to_string())
            .collect()
    }

    pub fn apply(&mut self, revision: &GraphDelta) -> DirectReachabilityUpdate {
        let mut update = DirectReachabilityUpdate::default();

        // Install new alternatives before removing old edges. This minimizes
        // transient retractions when an import is replaced in one revision.
        for ((source, target), diff) in &revision.edge_updates {
            if *diff > 0 {
                let source = self.intern(source);
                let target = self.intern(target);
                if self.insert_edge(source, target)
                    && self.reachable[source]
                    && !self.reachable[target]
                {
                    self.activate_from(target, source, &mut update);
                }
            }
        }
        for ((source, target), diff) in &revision.edge_updates {
            if *diff < 0 {
                let Some(&source) = self.indices.get(source.as_str()) else {
                    continue;
                };
                let Some(&target) = self.indices.get(target.as_str()) else {
                    continue;
                };
                if self.remove_edge(source, target) && self.parent[target] == Some(source) {
                    self.repair_detached_subtree(source, target, &mut update);
                }
            }
        }

        update
    }

    fn intern(&mut self, id: &str) -> usize {
        if let Some(&index) = self.indices.get(id) {
            return index;
        }
        let index = self.ids.len();
        let id = SharedModuleId::from(id);
        self.ids.push(id.clone());
        self.indices.insert(id, index);
        self.outgoing.push(Vec::new());
        self.incoming.push(Vec::new());
        self.reachable.push(false);
        self.parent.push(None);
        self.tree_children.push(Vec::new());
        self.subtree_marks.push(0);
        index
    }

    fn insert_edge(&mut self, source: usize, target: usize) -> bool {
        if self.outgoing[source].contains(&target) {
            return false;
        }
        self.outgoing[source].push(target);
        self.incoming[target].push(source);
        true
    }

    fn remove_edge(&mut self, source: usize, target: usize) -> bool {
        let Some(position) = self.outgoing[source]
            .iter()
            .position(|candidate| *candidate == target)
        else {
            return false;
        };
        self.outgoing[source].swap_remove(position);
        if let Some(position) = self.incoming[target]
            .iter()
            .position(|candidate| *candidate == source)
        {
            self.incoming[target].swap_remove(position);
        }
        true
    }

    fn recompute(&mut self) {
        self.reachable.fill(false);
        self.parent.fill(None);
        for children in &mut self.tree_children {
            children.clear();
        }
        self.reachable_count = 1;
        self.reachable[self.entry] = true;
        let mut queue = VecDeque::from([self.entry]);
        while let Some(source) = queue.pop_front() {
            for &target in &self.outgoing[source] {
                if self.reachable[target] {
                    continue;
                }
                self.reachable[target] = true;
                self.reachable_count += 1;
                self.parent[target] = Some(source);
                self.tree_children[source].push(target);
                queue.push_back(target);
            }
        }
    }

    fn activate_from(
        &mut self,
        target: usize,
        parent: usize,
        update: &mut DirectReachabilityUpdate,
    ) {
        self.set_reachable(target, true, update);
        self.parent[target] = Some(parent);
        self.tree_children[parent].push(target);
        let mut queue = VecDeque::from([target]);
        while let Some(source) = queue.pop_front() {
            for edge_index in 0..self.outgoing[source].len() {
                let target = self.outgoing[source][edge_index];
                if self.reachable[target] {
                    continue;
                }
                self.set_reachable(target, true, update);
                self.parent[target] = Some(source);
                self.tree_children[source].push(target);
                queue.push_back(target);
            }
        }
    }

    fn repair_detached_subtree(
        &mut self,
        old_parent: usize,
        root: usize,
        update: &mut DirectReachabilityUpdate,
    ) {
        if let Some(position) = self.tree_children[old_parent]
            .iter()
            .position(|child| *child == root)
        {
            self.tree_children[old_parent].swap_remove(position);
        }

        let mut subtree = Vec::new();
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            subtree.push(node);
            stack.extend(self.tree_children[node].iter().copied());
        }

        if subtree.len() * Self::RECOMPUTE_DENOMINATOR
            >= self.reachable_count * Self::RECOMPUTE_NUMERATOR
        {
            let before = self.reachable.clone();
            self.recompute();
            for (node, was_reachable) in before.into_iter().enumerate() {
                if was_reachable != self.reachable[node] {
                    self.record_change(node, self.reachable[node], update);
                }
            }
            update.used_full_recompute = true;
            return;
        }

        self.mark_epoch = self.mark_epoch.wrapping_add(1);
        if self.mark_epoch == 0 {
            self.subtree_marks.fill(0);
            self.mark_epoch = 1;
        }
        for &node in &subtree {
            self.subtree_marks[node] = self.mark_epoch;
            self.set_reachable(node, false, update);
            self.parent[node] = None;
            self.tree_children[node].clear();
        }

        let mut queue = VecDeque::new();
        for &node in &subtree {
            let external_parent = self.incoming[node]
                .iter()
                .copied()
                .find(|predecessor| self.reachable[*predecessor]);
            if let Some(parent) = external_parent {
                self.set_reachable(node, true, update);
                self.parent[node] = Some(parent);
                self.tree_children[parent].push(node);
                queue.push_back(node);
            }
        }

        while let Some(source) = queue.pop_front() {
            for edge_index in 0..self.outgoing[source].len() {
                let target = self.outgoing[source][edge_index];
                if self.subtree_marks[target] != self.mark_epoch || self.reachable[target] {
                    continue;
                }
                self.set_reachable(target, true, update);
                self.parent[target] = Some(source);
                self.tree_children[source].push(target);
                queue.push_back(target);
            }
        }
    }

    fn set_reachable(
        &mut self,
        node: usize,
        reachable: bool,
        update: &mut DirectReachabilityUpdate,
    ) {
        if self.reachable[node] == reachable {
            return;
        }
        self.reachable[node] = reachable;
        if reachable {
            self.reachable_count += 1;
        } else {
            self.reachable_count -= 1;
        }
        self.record_change(node, reachable, update);
    }

    fn record_change(&self, node: usize, reachable: bool, update: &mut DirectReachabilityUpdate) {
        let id = self.ids[node].as_ref();
        if reachable {
            if !update.removed.remove(id) {
                update.added.insert(id.to_owned());
            }
        } else if !update.added.remove(id) {
            update.removed.insert(id.to_owned());
        }
    }
}

fn load_uncached(
    resolver: &Resolver,
    resolution_cache: &ResolutionCache,
    path: &Path,
) -> Result<LoadedModule, String> {
    let id = path.to_string_lossy();
    // A loader (query, stylesheet, or asset) may claim this id before it is ever
    // read as JavaScript.
    if let Some(special) = load_special_module(&id, path) {
        let special = special?;
        return Ok(LoadedModule {
            hash: special.hash,
            dependencies: Vec::new(),
            pruned_imports: HashSet::new(),
            source: Arc::from(id.as_ref()),
            flat_module: special.flat_module,
            code: special.code,
            diagnostics: Vec::new(),
            assets: special.assets,
            css: special.css,
            externals: Vec::new(),
        });
    }
    let read_started = frontend_profile::start();
    let source = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    frontend_profile::finish(Phase::Read, read_started);
    let hash = content_hash(source.as_bytes());
    let transformed = transform_module(path, &source);
    let mut diagnostics = transformed
        .diagnostics
        .iter()
        .map(|diagnostic| format!("{}: {diagnostic}", path.display()))
        .collect::<Vec<_>>();
    let dependencies = resolve_dependencies(
        resolver,
        resolution_cache,
        path,
        &transformed.dependencies,
        &transformed.dependency_demands,
        &mut diagnostics,
    );
    Ok(LoadedModule {
        hash,
        dependencies: dependencies.dependencies,
        pruned_imports: dependencies.pruned_imports,
        source: Arc::from(source),
        flat_module: transformed.flat_module,
        code: transformed.code,
        diagnostics,
        assets: Vec::new(),
        css: None,
        externals: dependencies.externals,
    })
}

/// A non-JavaScript module produced by a loader: a query loader (`?url`, `?raw`),
/// a global stylesheet, or a default asset import. Callers wrap it into whichever
/// record they build (`ModuleState` or `LoadedModule`).
struct SpecialModule {
    hash: u64,
    code: String,
    flat_module: Option<FlatModule>,
    assets: Vec<AssetEmit>,
    css: Option<String>,
}

/// Loads a non-JavaScript module when a loader applies to `path`/`id`: a query
/// loader (`?url`, `?raw`), a global stylesheet (`.css`), or a default asset
/// import (image/font/SVG/...). Returns `None` for an ordinary JS/TS module,
/// which the normal read-and-transform path then handles.
fn load_special_module(id: &str, path: &Path) -> Option<Result<SpecialModule, String>> {
    let resource = ResourceId::parse(id);
    if resource.query.is_some() {
        return Some(synthesize_query_module(&resource));
    }
    if is_css_path(path) {
        return Some(load_stylesheet(path));
    }
    if is_asset_path(path) {
        return Some(synthesize_asset_url(path.to_path_buf()));
    }
    None
}

/// Builds the module for a query-bearing id. `?url` emits a content-hashed asset
/// and exports its URL; `?raw` inlines the file contents as a string.
/// Recognized-but-unimplemented loaders (`?tsr-split`) and unrecognized queries
/// produce a specific, actionable error rather than a misleading filesystem read
/// failure.
fn synthesize_query_module(resource: &ResourceId) -> Result<SpecialModule, String> {
    match resource.loader_kind() {
        Some(LoaderKind::Url) => synthesize_asset_url(PathBuf::from(&resource.path)),
        Some(LoaderKind::Raw) => synthesize_raw(Path::new(&resource.path)),
        Some(LoaderKind::TsrSplit) | None => Err(resource.unimplemented_loader_error()),
    }
}

/// A content-hashed asset module: copies the file into `assets/` and exports its
/// public URL as the default export. Used for both `?url` and default asset
/// imports (images, fonts, SVG, ...).
fn synthesize_asset_url(source_path: PathBuf) -> Result<SpecialModule, String> {
    let bytes = fs::read(&source_path)
        .map_err(|error| format!("cannot read asset {}: {error}", source_path.display()))?;
    let public_name = asset_public_name(&source_path, content_hash(&bytes));
    // A plain ES module exporting the asset URL, run through the real transformer
    // so it yields flat-linker code and export metadata like any hand-written
    // module.
    let synthetic = format!("export default {};\n", quote(&format!("/assets/{public_name}")));
    let transformed = transform_module(Path::new("diffpack-url-asset.js"), &synthetic);
    Ok(SpecialModule {
        hash: content_hash(transformed.code.as_bytes()),
        code: transformed.code,
        flat_module: transformed.flat_module,
        assets: vec![AssetEmit {
            source: source_path,
            public_name,
        }],
        css: None,
    })
}

/// A `?raw` module: the file's contents inlined as the default string export.
fn synthesize_raw(source_path: &Path) -> Result<SpecialModule, String> {
    let text = fs::read_to_string(source_path)
        .map_err(|error| format!("cannot read {}: {error}", source_path.display()))?;
    let synthetic = format!("export default {};\n", quote(&text));
    let transformed = transform_module(Path::new("diffpack-raw-asset.js"), &synthetic);
    Ok(SpecialModule {
        hash: content_hash(transformed.code.as_bytes()),
        code: transformed.code,
        flat_module: transformed.flat_module,
        assets: Vec::new(),
        css: None,
    })
}

/// A global stylesheet import: an empty JavaScript module (the import has no
/// bindings) whose text is extracted into the output stylesheet.
fn load_stylesheet(path: &Path) -> Result<SpecialModule, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    Ok(SpecialModule {
        hash: content_hash(text.as_bytes()),
        code: String::new(),
        flat_module: None,
        assets: Vec::new(),
        css: Some(text),
    })
}

/// Whether a resolved path is a plain global stylesheet (`import "./app.css"`).
fn is_css_path(path: &Path) -> bool {
    matches!(path.extension().and_then(|value| value.to_str()), Some("css"))
}

/// Whether a resolved path is a static asset imported for its URL by default
/// (images, fonts, SVG, media, and similar opaque files).
fn is_asset_path(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|value| value.to_str()),
        Some(
            "png" | "jpg" | "jpeg" | "gif" | "svg" | "webp" | "avif" | "ico" | "bmp" | "woff"
                | "woff2" | "ttf" | "otf" | "eot" | "mp4" | "webm" | "mp3" | "wav" | "ogg"
                | "pdf" | "wasm"
        )
    )
}

/// The content-hashed public filename for an asset, e.g. `app-1a2b3c4d5e6f7080.css`.
fn asset_public_name(path: &Path, hash: u64) -> String {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("asset");
    match path.extension().and_then(|value| value.to_str()) {
        Some(extension) => format!("{stem}-{hash:016x}.{extension}"),
        None => format!("{stem}-{hash:016x}"),
    }
}

/// Whether a specifier is external (not bundled): a Node built-in, addressed
/// either with the unambiguous `node:` prefix or as a bare builtin name. External
/// imports are left in the output for the runtime to resolve.
fn is_external_specifier(specifier: &str) -> bool {
    if let Some(builtin) = specifier.strip_prefix("node:") {
        // `node:test`, `node:fs/promises`, etc. The prefix alone is authoritative.
        return !builtin.is_empty();
    }
    let root = specifier.split('/').next().unwrap_or(specifier);
    matches!(
        root,
        "assert"
            | "async_hooks"
            | "buffer"
            | "child_process"
            | "cluster"
            | "console"
            | "constants"
            | "crypto"
            | "dgram"
            | "diagnostics_channel"
            | "dns"
            | "domain"
            | "events"
            | "fs"
            | "http"
            | "http2"
            | "https"
            | "inspector"
            | "module"
            | "net"
            | "os"
            | "path"
            | "perf_hooks"
            | "process"
            | "punycode"
            | "querystring"
            | "readline"
            | "repl"
            | "stream"
            | "string_decoder"
            | "sys"
            | "timers"
            | "tls"
            | "trace_events"
            | "tty"
            | "url"
            | "util"
            | "v8"
            | "vm"
            | "wasi"
            | "worker_threads"
            | "zlib"
    )
}

fn resolve_dependencies(
    resolver: &Resolver,
    resolution_cache: &ResolutionCache,
    path: &Path,
    dependency_specifiers: &[String],
    dependency_demands: &[DependencyDemand],
    diagnostics: &mut Vec<String>,
) -> ResolvedDependencies {
    let resolve_started = frontend_profile::start();
    let mut dependencies = Vec::with_capacity(dependency_specifiers.len());
    let mut pruned_imports = HashSet::new();
    let mut externals = Vec::new();
    let directory_cache = resolution_cache.directory(path);
    for specifier in dependency_specifiers {
        if dependencies
            .iter()
            .any(|(existing, _, _)| existing == specifier)
        {
            continue;
        }
        // An external (a Node built-in like `node:stream`) is not a graph module:
        // it is neither resolved nor bundled, and its `require(...)` is left in
        // place for the runtime to resolve. It is not a diagnostic.
        if is_external_specifier(specifier) {
            if !externals.iter().any(|existing| existing == specifier) {
                externals.push(specifier.clone());
            }
            continue;
        }
        match directory_cache.resolve(resolver, path, specifier) {
            Ok(resolved) => {
                let demand = dependency_demands
                    .iter()
                    .find(|demand| demand.specifier == *specifier)
                    .cloned()
                    .unwrap_or_else(|| DependencyDemand {
                        specifier: specifier.clone(),
                        all: true,
                        names: Vec::new(),
                        dynamic: false,
                    });
                if !demand.all
                    && demand.names.is_empty()
                    && resolved.side_effect_free
                    && !demand.dynamic
                {
                    pruned_imports.insert(specifier.clone());
                } else {
                    dependencies.push((specifier.clone(), resolved.id, demand));
                }
            }
            Err(error) => diagnostics.push(format!(
                "{}: cannot resolve {specifier:?}: {error}",
                path.display()
            )),
        }
    }
    frontend_profile::finish(Phase::Resolve, resolve_started);
    ResolvedDependencies {
        dependencies,
        pruned_imports,
        externals,
    }
}

struct ResolvedDependencies {
    dependencies: Vec<(String, SharedModuleId, DependencyDemand)>,
    pruned_imports: HashSet<String>,
    externals: Vec<String>,
}

struct RenderedBundle {
    code: String,
    mappings: Vec<ModuleMapping>,
}

struct ModuleMapping {
    dense_index: DenseModuleId,
    generated_line: u32,
    generated_lines: u32,
}

fn evaluate_fold_expression(
    expression: &FoldExpression,
    values: &HashMap<String, f64>,
) -> Option<f64> {
    match expression {
        FoldExpression::Number(bits) => Some(f64::from_bits(*bits)),
        FoldExpression::Reference(name) => values.get(name).copied(),
        FoldExpression::Add(left, right) => {
            Some(evaluate_fold_expression(left, values)? + evaluate_fold_expression(right, values)?)
        }
    }
}

fn display_fold_expression(expression: &FoldExpression) -> String {
    match expression {
        FoldExpression::Number(bits) => {
            format_javascript_number(f64::from_bits(*bits)).unwrap_or_else(|| "<non-finite>".into())
        }
        FoldExpression::Reference(name) => name.clone(),
        FoldExpression::Add(left, right) => format!(
            "({} + {})",
            display_fold_expression(left),
            display_fold_expression(right)
        ),
    }
}

fn format_javascript_number(value: f64) -> Option<String> {
    if value.is_nan() {
        return Some("NaN".into());
    }
    if value == f64::INFINITY {
        return Some("1/0".into());
    }
    if value == f64::NEG_INFINITY {
        return Some("-1/0".into());
    }
    if value == 0.0 && value.is_sign_negative() {
        return Some("-0".into());
    }
    value.is_finite().then(|| value.to_string())
}

fn path_with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_owned();
    value.push(suffix);
    PathBuf::from(value)
}

fn chunk_path(output: &Path, index: usize) -> Result<PathBuf, String> {
    let parent = output
        .parent()
        .ok_or_else(|| format!("output has no parent: {}", output.display()))?;
    let stem = output
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| format!("output filename is not UTF-8: {}", output.display()))?;
    let extension = output
        .extension()
        .and_then(|extension| extension.to_str())
        .map_or(String::new(), |extension| format!(".{extension}"));
    Ok(parent.join(format!("{stem}.chunk-{index}{extension}")))
}

/// Build-time configuration a plugin host contributes. Currently the resolver
/// aliases (specifier -> absolute target), such as TanStack's
/// `#tanstack-router-entry` -> `<app>/src/router.tsx`. Kept small and owned by
/// Rust; the host merely supplies the values.
#[derive(Debug, Clone, Default)]
pub struct BuildConfig {
    /// Ordered `(specifier, absolute_target)` alias pairs.
    pub aliases: Vec<(String, String)>,
    /// Environment resolve conditions (e.g. client `["module","browser",
    /// "production"]`, server `["node",...]`). This is what isolates client from
    /// server: browser conditions select packages' browser exports and exclude
    /// server-only code. Empty means the built-in default.
    pub conditions: Vec<String>,
}

fn resolve_options(config: &BuildConfig) -> ResolveOptions {
    // Without host-supplied conditions, keep the built-in default. With them (the
    // environment's browser/node conditions), keep `import`/`default` too so basic
    // ESM resolution still works.
    let condition_names = if config.conditions.is_empty() {
        vec!["import".into(), "module".into(), "default".into()]
    } else {
        let mut names = config.conditions.clone();
        for fallback in ["import", "default"] {
            if !names.iter().any(|name| name == fallback) {
                names.push(fallback.to_string());
            }
        }
        names
    };
    ResolveOptions {
        tsconfig: Some(TsconfigDiscovery::Auto),
        extensions: [".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".json"]
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
        condition_names,
        main_fields: vec!["module".into(), "main".into()],
        ..ResolveOptions::default()
    }
}

fn module_id(path: &Path) -> SharedModuleId {
    SharedModuleId::from(path.to_string_lossy().into_owned())
}

/// A module id built from a resolved filesystem path, re-attaching the loader
/// query and fragment from `resource`. When both are absent this is identical to
/// [`module_id`], so a plain `app.css` import and an `app.css?url` import become
/// distinct graph keys.
fn module_id_with_resource(path: &Path, resource: &ResourceId) -> SharedModuleId {
    if resource.query.is_none() && resource.fragment.is_none() {
        return module_id(path);
    }
    let reattached = ResourceId {
        path: path.to_string_lossy().into_owned(),
        query: resource.query.clone(),
        fragment: resource.fragment.clone(),
    };
    SharedModuleId::from(reattached.to_id())
}

fn content_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn quote(value: &str) -> String {
    serde_json::to_string(value).expect("serializing a JavaScript string cannot fail")
}

#[cfg(test)]
mod tests {
    use std::process::Command;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn bundles_typescript_dynamic_import_and_a_package_into_executable_javascript() {
        if Command::new("node").arg("--version").output().is_err() {
            return;
        }

        let directory = tempdir().unwrap();
        let package = directory.path().join("node_modules/tiny-package");
        fs::create_dir_all(&package).unwrap();
        fs::write(
            directory.path().join("entry.ts"),
            r#"
                import message from "tiny-package";
                import { add } from "./math.js";
                console.log(`${message}:${add(2, 3)}`);
                import("./lazy.js").then(({ lazy }) => console.log(lazy));
            "#,
        )
        .unwrap();
        fs::write(
            directory.path().join("math.ts"),
            "export const add = (a: number, b: number): number => a + b;",
        )
        .unwrap();
        fs::write(
            directory.path().join("lazy.js"),
            "export const lazy = 'lazy-loaded';",
        )
        .unwrap();
        fs::write(
            package.join("package.json"),
            r#"{"name":"tiny-package","type":"module","exports":"./index.js"}"#,
        )
        .unwrap();
        fs::write(package.join("index.js"), "export default 'package-ok';").unwrap();

        let entry = directory.path().join("entry.ts");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = Bundler::discover(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        assert_eq!(reachable.len(), 4);
        bundler.emit(&reachable, &output).unwrap();

        let executed = Command::new("node").arg(&output).output().unwrap();
        assert!(
            executed.status.success(),
            "{}",
            String::from_utf8_lossy(&executed.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&executed.stdout),
            "package-ok:5\nlazy-loaded\n"
        );
    }

    #[test]
    fn url_asset_import_emits_a_content_hashed_file_and_exports_its_public_url() {
        let directory = tempdir().unwrap();
        let css = ".brand { color: red; }\n";
        fs::write(directory.path().join("styles.css"), css).unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import url from './styles.css?url';\nconsole.log(url);\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        // The entry plus the distinct `styles.css?url` asset module.
        let reachable = bundler.reachable_modules_direct();
        assert_eq!(reachable.len(), 2, "{reachable:?}");
        bundler.emit(&reachable, &output).unwrap();

        // The bundle exports the asset's public URL, not the raw path.
        let bundle = fs::read_to_string(&output).unwrap();
        let url = bundle
            .lines()
            .find_map(|line| line.find("/assets/styles-").map(|start| &line[start..]))
            .and_then(|rest| rest.split('"').next())
            .expect("bundle should reference the hashed asset url");
        assert!(url.ends_with(".css"), "{url}");

        // The content-hashed asset file is copied next to the bundle with the
        // exact original bytes.
        let asset_name = url.trim_start_matches("/assets/");
        let asset_path = directory.path().join("dist/assets").join(asset_name);
        assert_eq!(fs::read_to_string(&asset_path).unwrap(), css);

        // A second, identical asset would hash to the same name (determinism).
        assert_eq!(asset_name, asset_public_name(Path::new("styles.css"), content_hash(css.as_bytes())));

        if Command::new("node").arg("--version").output().is_ok() {
            let executed = Command::new("node").arg(&output).output().unwrap();
            assert!(
                executed.status.success(),
                "{}",
                String::from_utf8_lossy(&executed.stderr)
            );
            assert_eq!(String::from_utf8_lossy(&executed.stdout), format!("{url}\n"));
        }
    }

    #[test]
    fn raw_import_inlines_the_file_contents_as_a_string() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("note.txt"), "hello from raw").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import raw from './note.txt?raw';\nconsole.log(raw);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();
        if Command::new("node").arg("--version").output().is_ok() {
            let executed = Command::new("node").arg(&output).output().unwrap();
            assert!(
                executed.status.success(),
                "{}",
                String::from_utf8_lossy(&executed.stderr)
            );
            assert_eq!(
                String::from_utf8_lossy(&executed.stdout),
                "hello from raw\n"
            );
        }
    }

    #[test]
    fn default_asset_import_emits_a_hashed_file_and_exports_its_url() {
        let directory = tempdir().unwrap();
        let svg = "<svg></svg>";
        fs::write(directory.path().join("logo.svg"), svg).unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import logo from './logo.svg';\nconsole.log(logo);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        assert_eq!(reachable.len(), 2, "{reachable:?}");
        bundler.emit(&reachable, &output).unwrap();

        let bundle = fs::read_to_string(&output).unwrap();
        let url = bundle
            .lines()
            .find_map(|line| line.find("/assets/logo-").map(|start| &line[start..]))
            .and_then(|rest| rest.split('"').next())
            .expect("bundle should reference the hashed asset url");
        assert!(url.ends_with(".svg"), "{url}");
        let asset_path = directory
            .path()
            .join("dist/assets")
            .join(url.trim_start_matches("/assets/"));
        assert_eq!(fs::read_to_string(&asset_path).unwrap(), svg);
    }

    #[test]
    fn an_unimplemented_loader_query_reports_a_specific_error() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("thing.js"), "export const x = 1;").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import c from './thing.js?tsr-split=component';\nconsole.log(c);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let error = match Bundler::discover_direct(&entry) {
            Ok(_) => panic!("an unimplemented loader must fail the build, not silently succeed"),
            Err(error) => error,
        };
        assert!(
            error.contains("loader `?tsr-split` is not yet implemented"),
            "{error}"
        );
        assert!(!error.contains("No such file or directory"), "{error}");
    }

    #[test]
    fn node_builtins_are_recognized_as_externals() {
        assert!(is_external_specifier("node:stream"));
        assert!(is_external_specifier("node:fs/promises"));
        assert!(is_external_specifier("fs"));
        assert!(is_external_specifier("async_hooks"));
        assert!(is_external_specifier("path/posix"));
        assert!(!is_external_specifier("react"));
        assert!(!is_external_specifier("./local"));
        assert!(!is_external_specifier("node:")); // empty builtin is not external
    }

    #[test]
    fn node_builtin_imports_are_left_external_and_run() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { basename } from 'node:path';\nimport { EOL } from 'node:os';\n\
             console.log(basename('/a/b/c.txt') + (EOL === '\\n' ? ':nl' : ':other'));\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        // Externals are neither resolved nor diagnosed nor added to the graph.
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        assert_eq!(reachable.len(), 1, "only the entry is a graph module: {reachable:?}");
        bundler.emit(&reachable, &output).unwrap();

        // The external require survives for the runtime to resolve.
        let bundle = fs::read_to_string(&output).unwrap();
        assert!(bundle.contains("require(\"node:path\")"), "{bundle}");

        if Command::new("node").arg("--version").output().is_ok() {
            let executed = Command::new("node").arg(&output).output().unwrap();
            assert!(
                executed.status.success(),
                "{}",
                String::from_utf8_lossy(&executed.stderr)
            );
            assert_eq!(String::from_utf8_lossy(&executed.stdout), "c.txt:nl\n");
        }
    }

    #[test]
    fn a_configured_alias_resolves_to_its_target() {
        // The shape of TanStack's `#tanstack-router-entry` -> app router: a bare
        // `#`-specifier the plugin host aliases to a real file.
        let directory = tempdir().unwrap();
        let router = directory.path().join("router.tsx");
        fs::write(&router, "export const router = 1;\n").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { router } from '#tanstack-router-entry';\nconsole.log(router);\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let config = BuildConfig {
            aliases: vec![(
                "#tanstack-router-entry".to_string(),
                router.to_string_lossy().into_owned(),
            )],
            ..BuildConfig::default()
        };
        let (bundler, update) = Bundler::discover_direct_with_config(&entry, &config).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);

        let reachable = bundler.reachable_modules_direct();
        assert_eq!(reachable.len(), 2, "{reachable:?}");
        assert!(
            reachable.iter().any(|id| id.contains("router.tsx")),
            "aliased import must resolve to the real router file: {reachable:?}"
        );
    }

    #[test]
    fn global_css_side_effect_imports_are_extracted_into_one_stylesheet() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("a.css"), ".a { color: red; }").unwrap();
        fs::write(directory.path().join("b.css"), ".b { color: blue; }").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './a.css';\nimport './b.css';\nconsole.log('ok');\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        // entry plus the two extracted stylesheets.
        let reachable = bundler.reachable_modules_direct();
        assert_eq!(reachable.len(), 3, "{reachable:?}");
        bundler.emit(&reachable, &output).unwrap();

        // Both stylesheets land in one extracted file, in import order.
        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        let a = css.find(".a { color: red; }").expect("a.css extracted");
        let b = css.find(".b { color: blue; }").expect("b.css extracted");
        assert!(a < b, "import order preserved: {css}");

        // The CSS is not left in the JavaScript bundle.
        let js = fs::read_to_string(&output).unwrap();
        assert!(!js.contains("color: red"), "{js}");

        if Command::new("node").arg("--version").output().is_ok() {
            let executed = Command::new("node").arg(&output).output().unwrap();
            assert!(
                executed.status.success(),
                "{}",
                String::from_utf8_lossy(&executed.stderr)
            );
            assert_eq!(String::from_utf8_lossy(&executed.stdout), "ok\n");
        }
    }

    #[test]
    fn rebuilds_only_the_changed_module_and_updates_live_reachability() {
        if Command::new("node").arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.js");
        let value = directory.path().join("value.ts");
        let output = directory.path().join("bundle.js");
        fs::write(
            &entry,
            "import { value } from './value.js'; console.log(value);",
        )
        .unwrap();
        fs::write(&value, "export const value: number = 1;").unwrap();

        let (mut bundler, _) = Bundler::discover(&entry).unwrap();
        let mut session = bundler.direct_reachability();
        let mut reachable = session.reachable_modules();
        bundler.emit(&reachable, &output).unwrap();
        assert_eq!(run_node(&output), "1\n");

        fs::write(&value, "export const value: number = 2;").unwrap();
        let update = bundler.rebuild_path(&value).unwrap();
        assert_eq!(update.transformed_modules, 1);
        assert_eq!(update.delta.changed.len(), 1);
        let result = session.apply(&update.delta);
        for removed in result.removed {
            reachable.remove(&removed);
        }
        reachable.extend(result.added);
        bundler.emit(&reachable, &output).unwrap();

        assert_eq!(run_node(&output), "2\n");
        assert_eq!(reachable.len(), 2);

        fs::write(&entry, "console.log('detached');").unwrap();
        let update = bundler.rebuild_path(&entry).unwrap();
        let result = session.apply(&update.delta);
        for removed in result.removed {
            reachable.remove(&removed);
        }
        reachable.extend(result.added);
        assert_eq!(reachable, bundler.reachable_modules_direct());
        assert_eq!(reachable.len(), 1);
    }

    #[test]
    fn resolves_typescript_path_aliases_from_the_nearest_tsconfig() {
        if Command::new("node").arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let source = directory.path().join("src");
        fs::create_dir_all(&source).unwrap();
        fs::write(
            directory.path().join("tsconfig.json"),
            r#"{"compilerOptions":{"paths":{"~/*":["./src/*"]}}}"#,
        )
        .unwrap();
        let entry = source.join("entry.ts");
        let output = directory.path().join("bundle.js");
        fs::write(
            &entry,
            "import { value } from '~/value'; console.log(value);",
        )
        .unwrap();
        fs::write(source.join("value.ts"), "export const value = 42;").unwrap();

        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        assert_eq!(reachable.len(), 2);
        bundler.emit(&reachable, &output).unwrap();
        assert_eq!(run_node(&output), "42\n");
    }

    #[test]
    fn minifies_foldable_values_across_modules_without_reparsing_output() {
        if Command::new("node").arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.js");
        let a = directory.path().join("a.js");
        let output = directory.path().join("bundle.js");
        fs::write(
            &entry,
            "import { a } from './a.js'; import { b } from './b.js'; console.log(a + b);",
        )
        .unwrap();
        fs::write(&a, "export const a = 1 + 2;").unwrap();
        fs::write(directory.path().join("b.js"), "export const b = 3;").unwrap();

        let (mut bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty());
        let reachable = bundler.reachable_modules_direct();
        bundler
            .emit_with_options(
                &reachable,
                &output,
                EmitOptions {
                    source_map: false,
                    minify: true,
                },
            )
            .unwrap();
        assert_eq!(fs::read_to_string(&output).unwrap(), "console.log(6);");
        assert_eq!(run_node(&output), "6\n");

        fs::write(&a, "export const a = Number('2');").unwrap();
        bundler.rebuild_path(&a).unwrap();
        bundler
            .emit_with_options(
                &reachable,
                &output,
                EmitOptions {
                    source_map: false,
                    minify: true,
                },
            )
            .unwrap();
        assert_eq!(run_node(&output), "5\n");
        assert_ne!(fs::read_to_string(&output).unwrap(), "console.log(5);");
    }

    #[test]
    fn direct_reachability_collects_a_detached_cycle_locally() {
        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.js");
        let a = directory.path().join("a.js");
        fs::write(
            &entry,
            concat!(
                "import './a.js';\n",
                "import './leaf-0.js';\n",
                "import './leaf-1.js';\n",
                "import './leaf-2.js';\n",
                "import './leaf-3.js';\n",
                "import './leaf-4.js';\n",
                "import './leaf-5.js';\n",
                "import './leaf-6.js';\n",
                "import './leaf-7.js';\n",
            ),
        )
        .unwrap();
        fs::write(&a, "import './b.js';").unwrap();
        fs::write(directory.path().join("b.js"), "import './a.js';").unwrap();
        for index in 0..8 {
            fs::write(
                directory.path().join(format!("leaf-{index}.js")),
                format!("export const leaf = {index};"),
            )
            .unwrap();
        }

        let (mut bundler, _) = Bundler::discover(&entry).unwrap();
        let mut direct = bundler.direct_reachability();
        fs::write(
            &entry,
            concat!(
                "import './leaf-0.js';\n",
                "import './leaf-1.js';\n",
                "import './leaf-2.js';\n",
                "import './leaf-3.js';\n",
                "import './leaf-4.js';\n",
                "import './leaf-5.js';\n",
                "import './leaf-6.js';\n",
                "import './leaf-7.js';\n",
            ),
        )
        .unwrap();

        let revision = bundler.rebuild_path(&entry).unwrap();
        let update = direct.apply(&revision.delta);

        assert_eq!(update.removed.len(), 2);
        assert!(!update.used_full_recompute);
        assert_eq!(
            direct.reachable_modules(),
            bundler.reachable_modules_direct()
        );
    }

    #[test]
    fn deleting_a_non_tree_edge_does_not_scan_or_change_reachability() {
        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.js");
        let a = directory.path().join("a.js");
        fs::write(&entry, "import './a.js'; import './b.js';").unwrap();
        fs::write(&a, "import './b.js';").unwrap();
        fs::write(directory.path().join("b.js"), "export const b = 1;").unwrap();

        let (mut bundler, _) = Bundler::discover(&entry).unwrap();
        let mut direct = bundler.direct_reachability();
        fs::write(&a, "export const a = 1;").unwrap();
        let revision = bundler.rebuild_path(&a).unwrap();
        let update = direct.apply(&revision.delta);

        assert!(update.added.is_empty());
        assert!(update.removed.is_empty());
        assert!(!update.used_full_recompute);
        assert_eq!(
            direct.reachable_modules(),
            bundler.reachable_modules_direct()
        );
    }

    #[test]
    fn direct_reachability_falls_back_for_a_large_detached_subtree() {
        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.js");
        fs::write(&entry, "import './a.js';").unwrap();
        fs::write(directory.path().join("a.js"), "import './b.js';").unwrap();
        fs::write(directory.path().join("b.js"), "export const b = 1;").unwrap();

        let (mut bundler, _) = Bundler::discover(&entry).unwrap();
        let mut direct = bundler.direct_reachability();
        fs::write(&entry, "export const entry = 1;").unwrap();
        let revision = bundler.rebuild_path(&entry).unwrap();
        let update = direct.apply(&revision.delta);

        assert!(update.used_full_recompute);
        assert_eq!(update.removed.len(), 2);
        assert_eq!(
            direct.reachable_modules(),
            bundler.reachable_modules_direct()
        );
    }

    #[test]
    fn emit_public_writes_a_client_layout_with_chunks_css_and_assets() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("style.css"), ".a { color: red; }").unwrap();
        fs::write(directory.path().join("logo.svg"), "<svg></svg>").unwrap();
        fs::write(
            directory.path().join("lazy.js"),
            "export const lazy = 'lazy';",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './style.css';\nimport logo from './logo.svg';\n\
             console.log(logo);\nimport('./lazy.js').then(({ lazy }) => console.log(lazy));\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output_root = directory.path().join(".diffpack-output");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let summary = bundler
            .emit_public(&reachable, &output_root, EmitOptions::default())
            .unwrap();

        // A main chunk plus the dynamically imported chunk.
        assert!(
            summary.javascript_files >= 2,
            "expected the entry chunk and a dynamic chunk: {summary:?}"
        );
        assert_eq!(summary.css_files, 1, "one extracted stylesheet: {summary:?}");
        assert_eq!(summary.asset_files, 1, "one hashed asset: {summary:?}");

        let public_dir = output_root.join("public");
        assert!(public_dir.join("client.js").is_file());
        assert!(public_dir.join("client.css").is_file());
        assert!(
            public_dir.join("assets").read_dir().unwrap().count() == 1,
            "the svg asset is copied under assets/"
        );
        // The summary counts exactly the files on disk.
        let on_disk = PublicBuildSummary::of(&public_dir).unwrap();
        assert_eq!(on_disk.javascript_files, summary.javascript_files);
        assert_eq!(on_disk.css_files, summary.css_files);
        assert_eq!(on_disk.asset_files, summary.asset_files);

        // A re-emit rebuilds `public/` from scratch: a file that would no longer
        // be produced does not linger.
        let stale = public_dir.join("stale.js");
        fs::write(&stale, "// stale").unwrap();
        bundler
            .emit_public(&reachable, &output_root, EmitOptions::default())
            .unwrap();
        assert!(!stale.exists(), "re-emit must clear stale output");
    }

    fn run_node(path: &Path) -> String {
        let output = Command::new("node").arg(path).output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap()
    }
}
