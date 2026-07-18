use std::collections::{BTreeSet, HashMap, VecDeque};
use std::fs;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use oxc_resolver::{ResolveOptions, Resolver, TsconfigDiscovery};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::dataflow::DeltaRevision;
use crate::frontend_profile::{self, Phase};
use crate::graph::ModuleId;
use crate::resource_id::ResourceId;
use crate::transform::transform_module;

type DenseModuleId = usize;
type SharedModuleId = Arc<str>;

#[derive(Debug, Clone)]
struct ModuleState {
    hash: u64,
    dependencies: Vec<(String, DenseModuleId)>,
    code: String,
}

struct LoadedModule {
    hash: u64,
    dependencies: Vec<(String, SharedModuleId)>,
    code: String,
    diagnostics: Vec<String>,
}

struct ResolutionCache {
    directories: [Mutex<HashMap<PathBuf, Arc<DirectoryResolutionCache>>>; 16],
}

struct DirectoryResolutionCache {
    directory: PathBuf,
    specifiers: [Mutex<HashMap<String, Result<SharedModuleId, String>>>; 64],
}

impl ResolutionCache {
    fn new() -> Self {
        Self {
            directories: std::array::from_fn(|_| Mutex::new(HashMap::new())),
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
    ) -> Result<SharedModuleId, String> {
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
        // A specifier may carry a loader query and/or fragment
        // (`app.css?url`, `route.tsx?tsr-split=component`). Only the path
        // component is a filesystem concern; the query is re-attached to the
        // resolved id and interpreted later, at load time. A query never causes
        // a resolve-time error.
        let resource = ResourceId::parse(specifier);
        let path_specifier = resource.path.as_str();
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
            Ok(resolved)
        } else {
            resolver
                .resolve_file(importer, path_specifier)
                .map_err(|error| error.to_string())
                .and_then(|resolution| {
                    let resolved = resolution.full_path();
                    if resolved.extension().and_then(|value| value.to_str()) == Some("node") {
                        Err(format!("native module {specifier:?} is not supported"))
                    } else {
                        Ok(module_id_with_resource(&resolved, &resource))
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
    pub delta: DeltaRevision,
    pub transformed_modules: usize,
    pub diagnostics: Vec<String>,
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
        Self::discover_inner(entry, true)
    }

    pub fn discover_direct(entry: &Path) -> Result<(Self, BuildUpdate), String> {
        Self::discover_inner(entry, false)
    }

    fn discover_inner(
        entry: &Path,
        record_initial_delta: bool,
    ) -> Result<(Self, BuildUpdate), String> {
        let entry_path = entry
            .canonicalize()
            .map_err(|error| format!("cannot open entry {}: {error}", entry.display()))?;
        let entry_id = module_id(&entry_path);
        let resolver = Resolver::new(resolve_options());
        let frontend_threads = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(4);
        let mut bundler = Self {
            entry: 0,
            ids: Vec::new(),
            indices: HashMap::new(),
            resolver,
            resolution_cache: ResolutionCache::new(),
            frontend_pool: ThreadPoolBuilder::new()
                .num_threads(frontend_threads)
                .thread_name(|index| format!("diffpack-frontend-{index}"))
                .build()
                .map_err(|error| format!("cannot create frontend worker pool: {error}"))?,
            modules: Vec::new(),
        };
        bundler.entry = bundler.intern(entry_id.clone());

        let mut delta = DeltaRevision {
            label: "initial-build".into(),
            ..DeltaRevision::default()
        };
        if record_initial_delta {
            delta.entry_updates.push((entry_id.to_string(), 1));
        }
        let mut diagnostics = Vec::new();
        let transformed_modules = bundler.discover_from(
            vec![entry_id],
            &mut delta,
            &mut diagnostics,
            record_initial_delta,
        )?;
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
                delta: DeltaRevision {
                    label: format!("ignored:{}", path.display()),
                    ..DeltaRevision::default()
                },
                transformed_modules: 0,
                diagnostics: Vec::new(),
            });
        };
        let Some(old) = self.modules[index].clone() else {
            return Ok(BuildUpdate {
                delta: DeltaRevision {
                    label: format!("ignored:{}", path.display()),
                    ..DeltaRevision::default()
                },
                transformed_modules: 0,
                diagnostics: Vec::new(),
            });
        };

        let mut delta = DeltaRevision {
            label: format!("change:{}", path.display()),
            ..DeltaRevision::default()
        };
        let mut diagnostics = Vec::new();
        if !path.is_file() {
            delta.module_updates.push(((id.to_string(), old.hash), -1));
            for (_, target) in &old.dependencies {
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
            delta.module_updates.push(((id.to_string(), old.hash), -1));
            delta.module_updates.push(((id.to_string(), new.hash), 1));
            delta.changed.insert(id.to_string());
        }
        let old_edges = old
            .dependencies
            .iter()
            .map(|(_, target)| target)
            .map(|target| (id.to_string(), self.ids[*target].to_string()))
            .collect::<BTreeSet<_>>();
        let new_edges = new
            .dependencies
            .iter()
            .map(|(_, target)| target)
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
            .map(|(_, target)| target)
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
        let parent = output
            .parent()
            .ok_or_else(|| format!("output has no parent: {}", output.display()))?;
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
        fs::write(output, self.render(reachable))
            .map_err(|error| format!("cannot write {}: {error}", output.display()))
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
        delta: &mut DeltaRevision,
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
                if record_delta {
                    delta
                        .module_updates
                        .push(((path.to_string(), loaded.hash), 1));
                }
                let mut dependencies = Vec::with_capacity(loaded.dependencies.len());
                for (specifier, target) in loaded.dependencies {
                    let target_index = self.intern(target.clone());
                    if record_delta {
                        delta
                            .edge_updates
                            .push(((path.to_string(), target.to_string()), 1));
                    }
                    if self.modules[target_index].is_none() {
                        frontier.insert(target.clone());
                    }
                    dependencies.push((specifier, target_index));
                }
                self.modules[source] = Some(ModuleState {
                    hash: loaded.hash,
                    dependencies,
                    code: loaded.code,
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
        // Guard against a query-bearing id reaching a raw file read. A watched
        // path is normally a bare file, but the id is the module identity and
        // must not be fed to `fs::read_to_string` verbatim when it carries a
        // loader query.
        let id = module_id(path);
        let resource = ResourceId::parse(&id);
        if resource.query.is_some() {
            return Err(resource.unimplemented_loader_error());
        }
        let path = Path::new(resource.path.as_str());
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
            diagnostics,
        );
        let dependencies = resolved_dependencies
            .into_iter()
            .map(|(specifier, target)| (specifier, self.intern(target)))
            .collect();

        Ok(ModuleState {
            hash,
            dependencies,
            code: transformed.code,
        })
    }

    fn render(&self, reachable: &BTreeSet<ModuleId>) -> String {
        let reachable = reachable
            .iter()
            .filter_map(|id| self.indices.get(id.as_str()).copied())
            .collect::<Vec<_>>();
        let mut runtime_indices = vec![None; self.ids.len()];
        for (runtime_index, &dense_index) in reachable.iter().enumerate() {
            runtime_indices[dense_index] = Some(runtime_index);
        }
        let fragments = reachable
            .par_iter()
            .enumerate()
            .filter_map(|(index, &dense_index)| {
                let module = self.modules[dense_index].as_ref()?;
                let module_fragment = format!(
                    "{index}:function(module,exports,require,__toESM,__export,__reExport,__import,__esmNamespace,__seal){{\n{}\n}},\n",
                    module.code
                );
                let mut map_fragment = format!("{index}:{{");
                for (specifier, target) in &module.dependencies {
                    let target = runtime_indices[*target]
                        .expect("a reachable module dependency must also be reachable");
                    map_fragment.push_str(&format!("{}:{target},", quote(specifier)));
                }
                map_fragment.push_str("},\n");
                Some((module_fragment, map_fragment))
            })
            .collect::<Vec<_>>();
        let mut modules = String::new();
        let mut maps = String::new();
        for (module, map) in fragments {
            modules.push_str(&module);
            maps.push_str(&map);
        }

        format!(
            r#"(()=>{{
"use strict";
const __modules={{{modules}}};
const __maps={{{maps}}};
const __cache=Object.create(null);
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
function __require(id){{
  if(__cache[id])return __cache[id].exports;
  const factory=__modules[id];
  if(!factory)throw new Error("Module is not in bundle: "+id);
  const module={{exports:{{}}}};
  __cache[id]=module;
  const require=specifier=>{{const target=__maps[id][specifier];if(target===undefined)throw new Error("Cannot resolve "+specifier+" from "+id);return __require(target);}};
  factory(module,module.exports,require,__toESM,__export,__reExport,__import,__esmNamespace,__seal);
  return module.exports;
}}
return __require({entry});
}})();
"#,
            entry = runtime_indices[self.entry].expect("entry module must be reachable")
        )
    }
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
            for (_, target) in &module.dependencies {
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

    pub fn apply(&mut self, revision: &DeltaRevision) -> DirectReachabilityUpdate {
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
    // The module id may carry a loader query. Split it off before touching the
    // filesystem: only the path component is a real file. A query-bearing id is
    // a loader concern, not a file read.
    let id = path.to_string_lossy();
    let resource = ResourceId::parse(&id);
    if resource.query.is_some() {
        return Err(resource.unimplemented_loader_error());
    }
    let path = Path::new(resource.path.as_str());
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
        &mut diagnostics,
    );
    Ok(LoadedModule {
        hash,
        dependencies,
        code: transformed.code,
        diagnostics,
    })
}

fn resolve_dependencies(
    resolver: &Resolver,
    resolution_cache: &ResolutionCache,
    path: &Path,
    dependency_specifiers: &[String],
    diagnostics: &mut Vec<String>,
) -> Vec<(String, SharedModuleId)> {
    let resolve_started = frontend_profile::start();
    let mut dependencies = Vec::with_capacity(dependency_specifiers.len());
    let directory_cache = resolution_cache.directory(path);
    for specifier in dependency_specifiers {
        if dependencies
            .iter()
            .any(|(existing, _)| existing == specifier)
        {
            continue;
        }
        match directory_cache.resolve(resolver, path, specifier) {
            Ok(resolved) => {
                dependencies.push((specifier.clone(), resolved));
            }
            Err(error) => diagnostics.push(format!(
                "{}: cannot resolve {specifier:?}: {error}",
                path.display()
            )),
        }
    }
    frontend_profile::finish(Phase::Resolve, resolve_started);
    dependencies
}

fn resolve_options() -> ResolveOptions {
    ResolveOptions {
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
        condition_names: vec!["import".into(), "module".into(), "default".into()],
        main_fields: vec!["module".into(), "main".into()],
        // Resolve TypeScript `paths` aliases (e.g. `~/*` -> `./src/*`) against
        // the nearest `tsconfig.json` to each importing file. `Auto` walks up
        // from the importer, so nested projects each get their own config with
        // no project-root plumbing. Aliases match on the path component only:
        // `DirectoryResolutionCache::resolve` passes the query-stripped
        // `path_specifier` to the resolver and `module_id_with_resource`
        // re-attaches any loader query to the resolved id.
        tsconfig: Some(TsconfigDiscovery::Auto),
        ..ResolveOptions::default()
    }
}

fn module_id(path: &Path) -> SharedModuleId {
    SharedModuleId::from(path.to_string_lossy().into_owned())
}

/// Builds a module id from a resolved filesystem path, re-attaching the loader
/// query and fragment carried by `resource`. When both are absent this is
/// identical to [`module_id`], so a plain `app.css` import and an
/// `app.css?url` import become distinct graph keys.
fn module_id_with_resource(path: &Path, resource: &ResourceId) -> SharedModuleId {
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

    use crate::run_delta_revisions;

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
        let result = run_delta_revisions(vec![update.delta]);
        let reachable = bundler.reachable_modules_direct();
        assert_eq!(result[0].reachable_facts, 4);
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
    fn rebuilds_only_the_changed_module_and_updates_the_live_dataflow() {
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

        let (mut bundler, initial) = Bundler::discover(&entry).unwrap();
        let session = crate::DeltaSession::new();
        let initial_result = session.apply(initial.delta).unwrap();
        let mut reachable = initial_result.added;
        bundler.emit(&reachable, &output).unwrap();
        assert_eq!(run_node(&output), "1\n");

        fs::write(&value, "export const value: number = 2;").unwrap();
        let update = bundler.rebuild_path(&value).unwrap();
        assert_eq!(update.transformed_modules, 1);
        assert_eq!(update.delta.changed.len(), 1);
        let result = session.apply(update.delta).unwrap();
        for removed in result.removed {
            reachable.remove(&removed);
        }
        reachable.extend(result.added);
        bundler.emit(&reachable, &output).unwrap();

        assert_eq!(run_node(&output), "2\n");
        assert_eq!(result.reachable_facts, 2);

        fs::write(&entry, "console.log('detached');").unwrap();
        let update = bundler.rebuild_path(&entry).unwrap();
        let result = session.apply(update.delta).unwrap();
        for removed in result.removed {
            reachable.remove(&removed);
        }
        reachable.extend(result.added);
        assert_eq!(reachable, bundler.reachable_modules_direct());
        assert_eq!(reachable.len(), 1);
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
    fn resolve_reattaches_the_query_to_the_module_id() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("app.css"), ".x {}").unwrap();
        let importer = directory.path().join("entry.tsx");
        fs::write(&importer, "").unwrap();

        let resolver = Resolver::new(resolve_options());
        let cache = ResolutionCache::new();
        let dir_cache = cache.directory(&importer);

        let resolved = dir_cache
            .resolve(&resolver, &importer, "./app.css?url")
            .unwrap();
        let resource = ResourceId::parse(&resolved);
        assert_eq!(resource.query.as_deref(), Some("url"));
        assert!(resource.fragment.is_none());
        assert!(resolved.ends_with("app.css?url"), "{resolved}");
        // The path component (query stripped) points at the real file.
        assert!(Path::new(resource.path.as_str()).is_file());

        // A plain import of the same file is a distinct module id: same path,
        // no query.
        let plain = dir_cache
            .resolve(&resolver, &importer, "./app.css")
            .unwrap();
        assert_ne!(plain, resolved);
        assert!(ResourceId::parse(&plain).query.is_none());
    }

    #[test]
    fn tsconfig_paths_alias_resolves_to_the_real_file() {
        // A `~/*` -> `./src/*` alias declared in the nearest tsconfig.json must
        // resolve on the current dataflow foundation via oxc_resolver's
        // `TsconfigDiscovery::Auto`.
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::write(
            root.join("tsconfig.json"),
            r#"{
                "compilerOptions": {
                    "baseUrl": ".",
                    "paths": { "~/*": ["./src/*"] }
                }
            }"#,
        )
        .unwrap();
        fs::create_dir_all(root.join("src/lib")).unwrap();
        let target = root.join("src/lib/thing.ts");
        fs::write(&target, "export const thing = 1;").unwrap();
        let importer = root.join("src/entry.ts");
        fs::write(&importer, "import { thing } from \"~/lib/thing\";").unwrap();

        let resolver = Resolver::new(resolve_options());
        let cache = ResolutionCache::new();
        let dir_cache = cache.directory(&importer);

        let resolved = dir_cache
            .resolve(&resolver, &importer, "~/lib/thing")
            .expect("tsconfig `paths` alias `~/lib/thing` should resolve");
        let resource = ResourceId::parse(&resolved);
        assert!(resource.query.is_none());
        assert_eq!(
            Path::new(resource.path.as_str()).canonicalize().unwrap(),
            target.canonicalize().unwrap(),
            "resolved {resolved} should point at the real src/lib/thing.ts",
        );
    }

    #[test]
    fn loading_a_query_bearing_id_yields_the_specific_unimplemented_loader_error() {
        let directory = tempdir().unwrap();
        let css = directory.path().join("app.css");
        fs::write(&css, ".x {}").unwrap();

        let resolver = Resolver::new(resolve_options());
        let cache = ResolutionCache::new();
        let id = format!("{}?url", css.display());
        let error = match load_uncached(&resolver, &cache, Path::new(&id)) {
            Ok(_) => panic!("expected a query-bearing id to fail loading"),
            Err(error) => error,
        };

        assert!(
            error.contains("loader `?url` is not yet implemented"),
            "{error}"
        );
        assert!(error.contains("requested for"), "{error}");
        // Crucially, not the old misleading filesystem crash.
        assert!(!error.contains("No such file or directory"), "{error}");
        assert!(!error.contains("cannot read"), "{error}");
    }

    #[test]
    fn loading_a_plain_id_reads_the_underlying_file() {
        let directory = tempdir().unwrap();
        let file = directory.path().join("mod.js");
        fs::write(&file, "export const value = 1;").unwrap();

        let resolver = Resolver::new(resolve_options());
        let cache = ResolutionCache::new();
        let loaded = load_uncached(&resolver, &cache, &file).unwrap();
        assert!(!loaded.code.is_empty());
        assert!(loaded.diagnostics.is_empty(), "{:?}", loaded.diagnostics);
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
