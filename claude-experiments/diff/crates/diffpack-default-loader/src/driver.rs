use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use diffpack_core::async_graph::{
    AsyncDependency, AsyncGraph, AsyncModules, detect as detect_async_modules,
    propagate as propagate_async_modules,
};
use diffpack_core::bundle::{
    ChunkGraph, ChunkPlan, EmitPlan, FlatRenderModule, IntegrationManifestChunk,
    IntegrationManifestGraph, IntegrationManifestModule, ModuleMapping, RenderCache,
    RenderKeyDependency, RenderKeyGraph, RenderKeyOptions, RenderedBundle, RuntimeRenderModule,
    assemble_runtime_literals, chunk_load_order, chunk_names, render_flat as render_flat_chunk,
    render_key, render_runtime_fragments, render_runtime_header, serialize_composed_source_map,
    serialize_readable_source_map, validate_mappings,
};
use diffpack_core::linker::{
    LinkDependency, LinkGraph, export_demands as derive_export_demands,
    live_modules as derive_live_modules,
};
use diffpack_core::minify::{
    chunk as minify_chunk_code, chunk_with_map as minify_chunk_code_with_map,
};
use diffpack_core::module_graph::{
    ModuleGraph, StaticGraphView, static_closure, static_execution_order,
};
use diffpack_core::runtime::{
    RuntimeIntegrationPolicy, RuntimePolicyModule, RuntimePolicyRequest, render_registry_runtime,
};
use diffpack_core::{
    BuildMode, Environment, HookContext, LoadRequest, Platform, ProviderPipeline, ResolveRequest,
    ResolveResult, SourceLanguage as ProviderLanguage,
};
use diffpack_default_loader::asset::{AssetEmission as AssetEmit, asset_variant_public_name};
pub use diffpack_default_loader::driver_config::{BuildConfig, DriverPolicies};
use diffpack_default_loader::loader::{self as loader_policy, LoaderKind};
use diffpack_default_loader::module::{ComponentSideEffects, SpecialModule};
use diffpack_default_loader::module_policy::SpecialModulePolicy;
pub use diffpack_default_loader::output::EmitSummary;
use diffpack_default_loader::output::{OutputIntegrationPolicy, write_if_changed};
use diffpack_default_loader::resolution_diagnostic::{
    host_provided_module_message, node_builtin_in_browser_message,
    optional_dependency_missing_message, specifier_resolves_two_ways_message,
};
use diffpack_default_loader::resolver::{
    ImportSyntax, ResolverConfig, Resolvers, resolve_worker_entries,
};
use diffpack_default_loader::resolver_policy::host_provided_scheme;
use diffpack_default_loader::source_policy::SourceIntegrationPolicy;
use diffpack_default_loader::tailwind_project::{
    ScanSkip, app_tailwind_theme_full, collect_glob_sources, collect_scan_sources,
    installed_tailwind_dir, report_tailwind_engine, tailwind_scan_root, tailwind_source_globs,
    warn_on_tailwind_version_drift,
};
use oxc_resolver::{ResolveError, Resolver, SideEffects};

pub use diffpack_core::CancelToken as EmitCancel;
use diffpack_core::compiler::{CompileRequest, ModuleCompiler};
use diffpack_core::diagnostic::from_transform as source_diagnostics;
pub use diffpack_core::diagnostic::{Diagnostic, DiagnosticKind, partition_diagnostics};
pub use diffpack_core::graph::{
    BuildUpdate, DirectReachability, DirectReachabilityUpdate, GraphDelta,
};
#[allow(unused_imports)]
use diffpack_core::source_map::ColumnEdit;
use diffpack_core::source_map::{
    MapOrigin, MapToken, ModuleMapLookup, ModuleSourceMap, ResolvedMinifiedToken,
    resolve_minified_token,
};
use diffpack_core::tree_shake::Demand as ExportDemand;
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

use diffpack_core::frontend_profile::{self, Phase};
use diffpack_core::resource_id::ResourceId;
use diffpack_core::transform::{DependencyDemand, FlatModule, ModuleLiveness, Target};

pub type ModuleId = String;
pub type DenseModuleId = usize;
type SharedModuleId = Arc<str>;

#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct ModuleState {
    /// Identity of the module's SOURCE input: for a regular module the hash of the
    /// file bytes (drives the "skip re-transform if unchanged" fast path and the
    /// rebuild source-change check); for a virtual/special module the hash of its
    /// synthesized code. NOT the render-cache key — see `code_hash`.
    hash: u64,
    /// Identity of the module's TRANSFORMED output — the bytes that actually land
    /// in a rendered chunk. The per-chunk render cache keys on this, so a source
    /// edit that leaves the transformed output unchanged (e.g. editing a route
    /// component whose body was already split into its own chunk, leaving the
    /// reference module byte-identical) reuses the chunk instead of needlessly
    /// re-rendering it.
    code_hash: u64,
    dependencies: Vec<(String, DenseModuleId, DependencyDemand)>,
    pruned_imports: HashSet<String>,
    source: SharedModuleId,
    flat_module: Option<FlatModule>,
    code: String,
    assets: Vec<AssetEmit>,
    provider_assets: Vec<diffpack_core::EmittedAsset>,
    /// Stylesheet text contributed by a global CSS side-effect import
    /// (`import "./app.css"`). Extracted and concatenated into the output
    /// stylesheet in module execution order; `None` for a normal JS module.
    css: Option<String>,
    /// Physical CSS files (other than this module's own path) whose content was
    /// INLINED into this module's `css` — a media-qualified `@import`'s nested
    /// imports. An edit to any of them must re-derive this module
    /// ([`Bundler::rebuild_path`] scans for dependents); empty everywhere else.
    css_source_files: Vec<PathBuf>,
    /// Remote/absolute `@import` statements this module's CSS carried
    /// (`@import url(https://...)`). Hoisted verbatim, deduped, to the top of
    /// the emitted stylesheet by [`Bundler::emit_css`], because an `@import` is
    /// only valid before all rules.
    css_external_imports: Vec<String>,
    /// External specifiers (Node built-ins) this module imports. Left in the
    /// output for the runtime to resolve; a module with externals renders through
    /// the runtime path, since the flat path cannot bind an external.
    externals: Vec<String>,
    /// Whether this module's nearest `package.json` authorizes dropping it when
    /// none of its exports are used (`sideEffects:false`, or a `sideEffects` glob
    /// list it does not match). `false` — the conservative default for the app's
    /// own code, any package without the flag, and every synthesized module — means
    /// the module is always kept when reachable. Consulted only by the export-level
    /// dead-module elimination pass ([`Bundler::live_modules`]); never affects the
    /// incremental reachability index.
    droppable: bool,
    /// The module's export/import structure for export-level liveness (which of
    /// its exports forward an imported binding vs which imports are used in real
    /// code). Empty for synthesized modules, which fall back to treating every
    /// dependency as a body use.
    liveness: ModuleLiveness,
    /// The module `await`s at top level (only representable in flat ESM output;
    /// the emit hard-errors otherwise). `false` for every synthesized module.
    uses_top_level_await: bool,
    /// The module references `import.meta` (valid in ESM output, a syntax error
    /// in CommonJS output). `false` for every synthesized module.
    /// The module freely references a CommonJS ambient (`exports`, `module`,
    /// ...), so it must render through the factory runtime in ESM output.
    uses_cjs_globals: bool,
    /// The module freely references `__dirname`/`__filename`. A browser target
    /// has no location to derive them from, so its factory defines them.
    uses_dirname: bool,
    /// Module-worker entries: `(placeholder_key, resolved_entry_path)`.
    workers: Vec<(String, PathBuf)>,
    /// The module's REAL source map over `code`, as the Oxc printer emitted it.
    /// `None` unless the build asked for source maps, and `None` for a module
    /// whose code was synthesized rather than printed from an AST — those regions
    /// stay honestly UNMAPPED. See [`crate::source_map`].
    map: Option<ModuleSourceMap>,
}

struct LoadedModule {
    hash: u64,
    code_hash: u64,
    dependencies: Vec<(String, SharedModuleId, DependencyDemand)>,
    pruned_imports: HashSet<String>,
    source: SharedModuleId,
    flat_module: Option<FlatModule>,
    code: String,
    diagnostics: Vec<Diagnostic>,
    assets: Vec<AssetEmit>,
    provider_assets: Vec<diffpack_core::EmittedAsset>,
    css: Option<String>,
    css_source_files: Vec<PathBuf>,
    css_external_imports: Vec<String>,
    externals: Vec<String>,
    droppable: bool,
    liveness: ModuleLiveness,
    uses_top_level_await: bool,
    uses_cjs_globals: bool,
    uses_dirname: bool,
    /// The module's REAL source map over `code`; see [`ModuleState::map`].
    map: Option<ModuleSourceMap>,
    /// Module-worker entries this module creates: `(placeholder_key,
    /// resolved_entry_path)`. Emitted as self-contained bundles under
    /// `assets/`; the key ties the code placeholder to the emitted file.
    workers: Vec<(String, PathBuf)>,
}

struct AsyncGraphView<'a> {
    modules: &'a [Option<ModuleState>],
    ids: &'a [SharedModuleId],
    runtime_ids: Option<&'a [Option<usize>]>,
}

struct AsyncDependencyIter<'a> {
    dependencies: std::slice::Iter<'a, (String, DenseModuleId, DependencyDemand)>,
    pruned: &'a HashSet<String>,
}

struct StaticModuleGraphView<'a> {
    graph: &'a ModuleGraph<ModuleState>,
}

struct LinkModuleGraphView<'a> {
    graph: &'a ModuleGraph<ModuleState>,
}

struct ModuleMapView<'a> {
    graph: &'a ModuleGraph<ModuleState>,
}

impl ModuleMapLookup for ModuleMapView<'_> {
    fn module_map(&self, module: DenseModuleId) -> Option<(&ModuleSourceMap, &Arc<str>)> {
        let module = self.graph.modules.get(module)?.as_ref()?;
        Some((module.map.as_ref()?, &module.source))
    }
}

fn link_dependency(dependency: &(String, DenseModuleId, DependencyDemand)) -> LinkDependency<'_> {
    LinkDependency {
        specifier: &dependency.0,
        target: dependency.1,
        demand: &dependency.2,
    }
}

impl LinkGraph for LinkModuleGraphView<'_> {
    type Dependencies<'a>
        = std::iter::Map<
        std::slice::Iter<'a, (String, DenseModuleId, DependencyDemand)>,
        fn(&'a (String, DenseModuleId, DependencyDemand)) -> LinkDependency<'a>,
    >
    where
        Self: 'a;

    fn module_count(&self) -> usize {
        self.graph.modules.len()
    }

    fn entry(&self) -> DenseModuleId {
        self.graph.entry
    }

    fn present(&self, module: DenseModuleId) -> bool {
        self.graph.modules.get(module).is_some_and(Option::is_some)
    }

    fn droppable(&self, module: DenseModuleId) -> bool {
        self.graph.modules[module]
            .as_ref()
            .is_some_and(|module| module.droppable)
    }

    fn liveness(&self, module: DenseModuleId) -> Option<&ModuleLiveness> {
        self.graph.modules[module]
            .as_ref()
            .map(|module| &module.liveness)
    }

    fn dependencies(&self, module: DenseModuleId) -> Self::Dependencies<'_> {
        self.graph.modules[module]
            .as_ref()
            .expect("present link graph module")
            .dependencies
            .iter()
            .map(link_dependency)
    }
}

impl StaticGraphView for StaticModuleGraphView<'_> {
    type Dependencies<'a>
        = std::iter::FilterMap<
        std::slice::Iter<'a, (String, DenseModuleId, DependencyDemand)>,
        fn(&'a (String, DenseModuleId, DependencyDemand)) -> Option<DenseModuleId>,
    >
    where
        Self: 'a;

    fn module_id(&self, module: DenseModuleId) -> &str {
        &self.graph.ids[module]
    }

    fn present(&self, module: DenseModuleId) -> bool {
        self.graph.modules.get(module).is_some_and(Option::is_some)
    }

    fn dependencies(&self, module: DenseModuleId) -> Self::Dependencies<'_> {
        fn static_target(
            dependency: &(String, DenseModuleId, DependencyDemand),
        ) -> Option<DenseModuleId> {
            (!dependency.2.deferred()).then_some(dependency.1)
        }
        self.graph.modules[module]
            .as_ref()
            .expect("present static graph module")
            .dependencies
            .iter()
            .filter_map(static_target)
    }
}

impl<'a> Iterator for AsyncDependencyIter<'a> {
    type Item = AsyncDependency<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.dependencies
            .next()
            .map(|(specifier, target, _)| AsyncDependency {
                specifier,
                target: *target,
                pruned: self.pruned.contains(specifier),
            })
    }
}

impl AsyncGraph for AsyncGraphView<'_> {
    type Dependencies<'a>
        = AsyncDependencyIter<'a>
    where
        Self: 'a;

    fn module_count(&self) -> usize {
        self.modules.len()
    }
    fn id(&self, module: usize) -> &str {
        &self.ids[module]
    }
    fn emitted(&self, module: usize) -> bool {
        self.modules.get(module).is_some_and(Option::is_some)
            && self
                .runtime_ids
                .is_none_or(|runtime_ids| runtime_ids.get(module).is_some_and(Option::is_some))
    }
    fn uses_top_level_await(&self, module: usize) -> bool {
        self.modules[module]
            .as_ref()
            .is_some_and(|module| module.uses_top_level_await)
    }
    fn code(&self, module: usize) -> &str {
        &self.modules[module]
            .as_ref()
            .expect("emitted module exists")
            .code
    }
    fn dependencies(&self, module: usize) -> Self::Dependencies<'_> {
        let module = self.modules[module]
            .as_ref()
            .expect("emitted module exists");
        AsyncDependencyIter {
            dependencies: module.dependencies.iter(),
            pruned: &module.pruned_imports,
        }
    }
}

struct ResolutionCache {
    providers: ProviderPipeline,
    provider_environment: Environment,
    provider_root: Arc<Path>,
    directories: [Mutex<HashMap<PathBuf, Arc<DirectoryResolutionCache>>>; 16],
    /// Plugin-host aliases: `(specifier, absolute_target)` applied as an exact-
    /// match rewrite before the standards-aware resolver runs. Shared read-only,
    /// so cheap to clone into each directory cache.
    aliases: Arc<Vec<(String, PathBuf)>>,
    /// Build-generated virtual modules keyed by specifier. A matching specifier
    /// resolves to itself and loads from the recorded source rather than the
    /// filesystem.
    virtual_modules: Arc<HashMap<String, String>>,
    /// Vite `import.meta.env` values, when opted in. Shared read-only to both the
    /// serial and parallel module-load paths, which apply the rewrite to a
    /// module's source before it is transformed. `None` leaves `import.meta.env`
    /// untouched (generic bundling).
    source_policy: Arc<dyn SourceIntegrationPolicy>,
    /// Public base for minted asset URLs (always `/`-terminated).
    base: Arc<str>,
    /// A REAL directory (the entry module's directory) from which a build-generated
    /// virtual module's bare-package imports are resolved. A virtual module's id is
    /// a synthetic specifier (`#diffpack-call-server`), not a filesystem path, so it
    /// has no directory to walk `node_modules` up from — resolving
    /// `react-server-dom-webpack/client` off it fails. Virtual modules resolve their
    /// dependencies as if located here, so bare packages resolve against the project.
    virtual_import_base: Arc<Path>,
    /// Vite's `assetsInlineLimit` (0 = off).
    asset_inline_limit: usize,
    /// SCSS compile options (Vite `additionalData` + project root), threaded
    /// to the `.scss` loaders.
    scss: Arc<diffpack_default_loader::sass::ScssOptions>,
    /// How default raster-image imports materialize (Vite bare-URL string vs the
    /// Next static-import object shape). Threaded to `synthesize_asset_url`.
    image_import_shape: ImageImportShape,
    /// Less/Stylus + PostCSS wiring, threaded to the CSS loaders.
    css_preprocess: CssPreprocess,
    /// The project's JSX-extension rule, threaded to the module transform on both
    /// the serial ([`Bundler::load_module`]) and parallel ([`load_uncached`]) load
    /// paths — the one parse whose diagnostics the build reports.
    jsx_extensions: diffpack_core::parser::JsxExtensions,
    /// The BUILD's JSX lowering settings (`vite.config`'s `esbuild.*` / `oxc.jsx`).
    /// Layered over each file's owning tsconfig by [`jsx_config_for`] on both load
    /// paths; empty (the default) leaves the tsconfig — and, failing that, oxc's
    /// react-automatic default — in charge.
    jsx: Arc<diffpack_core::transform::JsxConfig>,
    /// Which directories the object form of `package.json`'s `browser` field can
    /// rewrite files in. Consulted before the relative-path fast path, which would
    /// otherwise answer with the file the field replaces.
    browser_field: Arc<BrowserFieldMap>,
    /// Package names a SERVER graph must not bundle — Next's
    /// `serverExternalPackages`. See [`ResolutionCache::is_external_package`].
    external_packages: Arc<Vec<String>>,
}

/// The module id every `"browser": { "…": false }` entry resolves to. A single
/// shared id, because every such module is the same empty module — one copy in
/// the bundle, however many packages exclude something.
const BROWSER_EXCLUDED_MODULE_ID: &str = "diffpack-browser-excluded:v";

/// The source of [`BROWSER_EXCLUDED_MODULE_ID`]. CommonJS, matching what webpack
/// substitutes: the importer sees an object with no properties, so a property
/// read yields `undefined` rather than a missing-export build failure. The
/// package asked for exactly this by writing `false`.
const BROWSER_EXCLUDED_MODULE_SOURCE: &str = "module.exports = {};\n";

/// Which directories a `package.json` `browser` field in its OBJECT form governs.
///
/// The object form remaps a package's own internal modules by package-relative
/// path (`{"./lib/adapters/http.js": "./lib/helpers/null.js"}`), so a plain
/// `import './http.js'` from inside that package must NOT be answered by simply
/// joining the path — the browser build is supposed to get the replacement. The
/// authoritative resolver knows this; the relative fast path in
/// [`DirectoryResolutionCache::resolve`] does not, so it has to ask here first.
///
/// Answers are memoized per directory: the walk up to the nearest `package.json`
/// happens once per directory, no matter how many specifiers resolve from it.
use diffpack_default_loader::browser_field::BrowserFieldMap;

/// Which syntax reaches a specifier, and so which export conditions resolve it.
///
/// `package.json`'s `exports` is a map keyed by condition, and `import` and
/// `require` are different keys pointing at different files in almost every
/// dual-published package. Resolving both the same way is not an approximation,
/// it hands back the wrong module: `pg/lib/index.js` does `require('pg-pool')`
/// and then `class Pool extends …`, and pg-pool's ESM entry is a Module
/// namespace object, which is not a constructor.
struct DirectoryResolutionCache {
    providers: ProviderPipeline,
    provider_environment: Environment,
    provider_root: Arc<Path>,
    directory: PathBuf,
    specifiers: [Mutex<HashMap<String, Result<ResolvedModule, String>>>; 64],
    /// The same cache for `require(...)` call sites, which resolve under
    /// different export conditions and so can legitimately answer differently
    /// for the identical specifier in the identical directory. Kept separate
    /// rather than folded into a `(syntax, specifier)` key so the overwhelmingly
    /// common ESM lookup still hashes a borrowed `&str` with no allocation.
    common_js_specifiers: [Mutex<HashMap<String, Result<ResolvedModule, String>>>; 16],
    aliases: Arc<Vec<(String, PathBuf)>>,
    virtual_modules: Arc<HashMap<String, String>>,
    /// The project root, when the build has one. A root-absolute specifier
    /// (`import icons from "/icons.svg"`) is resolved against it, and against its
    /// `public/` directory — see [`DirectoryResolutionCache::resolve_root_absolute`].
    root: Option<Arc<Path>>,
    /// Shared with the owning [`ResolutionCache`]: which directories an
    /// object-form `browser` field governs, so the relative fast path can stand
    /// aside where it would answer with the wrong file.
    browser_field: Arc<BrowserFieldMap>,
}

#[derive(Clone)]
struct ResolvedModule {
    id: SharedModuleId,
    side_effect_free: bool,
    provider_external: bool,
}

fn provider_messages(
    id: &str,
    messages: Vec<diffpack_core::ProviderMessage>,
) -> Vec<diffpack_core::Diagnostic> {
    messages
        .into_iter()
        .map(|message| diffpack_core::Diagnostic {
            kind: diffpack_core::DiagnosticKind::Source {
                fatal: message.fatal,
            },
            message: format!("provider diagnostic for {id}: {}", message.message),
        })
        .collect()
}

impl ResolutionCache {
    fn transform_external_source(
        &self,
        id: &str,
        code: &str,
        language: diffpack_core::transform::SourceLanguage,
        target: Target,
    ) -> Result<
        (
            String,
            diffpack_core::transform::SourceLanguage,
            Vec<diffpack_core::EmittedAsset>,
            Vec<PathBuf>,
            Vec<diffpack_core::Diagnostic>,
        ),
        String,
    > {
        let provider_language = match language {
            diffpack_core::transform::SourceLanguage::JavaScript => ProviderLanguage::JavaScript,
            diffpack_core::transform::SourceLanguage::TypeScript => ProviderLanguage::TypeScript,
            diffpack_core::transform::SourceLanguage::FromPath => match Path::new(id)
                .extension()
                .and_then(|extension| extension.to_str())
            {
                Some("ts" | "tsx" | "mts" | "cts") => ProviderLanguage::TypeScript,
                _ => ProviderLanguage::JavaScript,
            },
        };
        let environment = Environment {
            name: match target {
                Target::Client => "client",
                Target::Server => "server",
                Target::IsolatedServer => "react-server",
            }
            .into(),
            platform: if target == Target::Client {
                Platform::Browser
            } else {
                Platform::Node
            },
            mode: self.provider_environment.mode,
        };
        let source = diffpack_core::LoadedSource {
            code: code.as_bytes().to_vec(),
            language: provider_language,
            source_map: None,
            watch_files: Vec::new(),
            diagnostics: Vec::new(),
        };
        let (source, assets) = self
            .providers
            .transform(
                id,
                source,
                HookContext {
                    environment: &environment,
                    project_root: &self.provider_root,
                },
            )
            .map_err(|diagnostic| {
                diagnostic
                    .provider
                    .map_or(diagnostic.message.clone(), |provider| {
                        format!("provider {provider}: {}", diagnostic.message)
                    })
            })?;
        if source.source_map.is_some() {
            return Err(format!(
                "provider transform for {id:?} returned a source map, but provider map composition is not wired into chunk maps yet"
            ));
        }
        let language = if source.language == provider_language {
            language
        } else {
            match source.language {
                ProviderLanguage::JavaScript => {
                    diffpack_core::transform::SourceLanguage::JavaScript
                }
                ProviderLanguage::TypeScript => {
                    diffpack_core::transform::SourceLanguage::TypeScript
                }
                other => {
                    return Err(format!(
                        "provider transform for {id:?} returned unsupported language {other:?}"
                    ));
                }
            }
        };
        let code = String::from_utf8(source.code).map_err(|error| {
            format!("provider transform for {id:?} returned non-UTF-8: {error}")
        })?;
        let diagnostics = provider_messages(id, source.diagnostics);
        Ok((code, language, assets, source.watch_files, diagnostics))
    }

    fn provider_source(
        &self,
        id: &str,
        target: Target,
    ) -> Result<
        Option<(
            String,
            diffpack_core::transform::SourceLanguage,
            Vec<diffpack_core::EmittedAsset>,
            Vec<PathBuf>,
            Vec<diffpack_core::Diagnostic>,
        )>,
        String,
    > {
        let environment = Environment {
            name: match target {
                Target::Client => "client",
                Target::Server => "server",
                Target::IsolatedServer => "react-server",
            }
            .to_string(),
            platform: if target == Target::Client {
                Platform::Browser
            } else {
                Platform::Node
            },
            mode: self.provider_environment.mode,
        };
        let context = HookContext {
            environment: &environment,
            project_root: &self.provider_root,
        };
        let request = LoadRequest {
            id,
            context: context.clone(),
        };
        let Some(source) = self.providers.load(request).map_err(|diagnostic| {
            diagnostic
                .provider
                .map_or(diagnostic.message.clone(), |provider| {
                    format!("provider {provider}: {}", diagnostic.message)
                })
        })?
        else {
            return Ok(None);
        };
        let (source, emitted_assets) =
            self.providers
                .transform(id, source, context)
                .map_err(|diagnostic| {
                    diagnostic
                        .provider
                        .map_or(diagnostic.message.clone(), |provider| {
                            format!("provider {provider}: {}", diagnostic.message)
                        })
                })?;
        let language = match source.language {
            ProviderLanguage::JavaScript => diffpack_core::transform::SourceLanguage::JavaScript,
            ProviderLanguage::TypeScript => diffpack_core::transform::SourceLanguage::TypeScript,
            other => {
                return Err(format!(
                    "provider source for {id:?} has unsupported language {other:?}; the external source seam currently accepts JavaScript or TypeScript"
                ));
            }
        };
        if source.source_map.is_some() {
            return Err(format!(
                "provider source for {id:?} includes a source map, but provider map composition is not wired into chunk maps yet"
            ));
        }
        let watch_files = source.watch_files;
        let code = String::from_utf8(source.code)
            .map_err(|error| format!("provider source for {id:?} is not UTF-8: {error}"))?;
        let diagnostics = provider_messages(id, source.diagnostics);
        Ok(Some((
            code,
            language,
            emitted_assets,
            watch_files,
            diagnostics,
        )))
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        providers: ProviderPipeline,
        aliases: Vec<(String, PathBuf)>,
        virtual_modules: Vec<(String, String)>,
        source_policy: Arc<dyn SourceIntegrationPolicy>,
        base: &str,
        virtual_import_base: PathBuf,
        asset_inline_limit: usize,
        scss: diffpack_default_loader::sass::ScssOptions,
        image_import_shape: ImageImportShape,
        css_preprocess: CssPreprocess,
        jsx_extensions: diffpack_core::parser::JsxExtensions,
        jsx: diffpack_core::transform::JsxConfig,
        honors_browser_field: bool,
        provider_mode: BuildMode,
        external_packages: Vec<String>,
    ) -> Self {
        let provider_root: Arc<Path> = Arc::from(virtual_import_base.as_path());
        Self {
            providers,
            provider_environment: Environment {
                name: if honors_browser_field {
                    "client"
                } else {
                    "server"
                }
                .to_string(),
                platform: if honors_browser_field {
                    Platform::Browser
                } else {
                    Platform::Node
                },
                mode: provider_mode,
            },
            provider_root,
            external_packages: Arc::new(external_packages),
            browser_field: Arc::new(BrowserFieldMap::new(honors_browser_field)),
            directories: std::array::from_fn(|_| Mutex::new(HashMap::new())),
            aliases: Arc::new(aliases),
            virtual_modules: Arc::new(virtual_modules.into_iter().collect()),
            source_policy,
            base: Arc::from(base),
            virtual_import_base: Arc::from(virtual_import_base.as_path()),
            asset_inline_limit,
            scss: Arc::new(scss),
            image_import_shape,
            css_preprocess,
            jsx_extensions,
            jsx: Arc::new(jsx),
        }
    }

    /// Whether `specifier` names a package the build was told to leave OUT of a server
    /// bundle (Next's `serverExternalPackages`, and its older
    /// `experimental.serverComponentsExternalPackages` spelling).
    ///
    /// A package lands on that list when bundling it is wrong or impossible: it loads
    /// native addons, `require`s something optional that may not be installed, or reads
    /// files relative to its own location. cal.com lists eight, among them `rest-facade`
    /// — whose `require('superagent-proxy')` sits behind an `if (options.proxy)` and
    /// resolves to a package that is deliberately not installed. Bundling it turns a
    /// branch the app never takes into a fatal build error.
    ///
    /// Subpaths count: listing `jose` also externalizes `jose/errors`, matching Next.
    fn is_external_package(&self, specifier: &str) -> bool {
        self.external_packages.iter().any(|package| {
            specifier == package
                || (specifier.len() > package.len()
                    && specifier.starts_with(package.as_str())
                    && specifier.as_bytes()[package.len()] == b'/')
        })
    }

    /// The source of a build-generated virtual module for this id, if one is
    /// registered.
    fn virtual_module_source(&self, id: &str) -> Option<&str> {
        if id == BROWSER_EXCLUDED_MODULE_ID {
            return Some(BROWSER_EXCLUDED_MODULE_SOURCE);
        }
        // A runtime helper a lowering calls (`__decorate`, ...). Embedded in the
        // binary, so it is the same helper in every app and needs no install.
        if let Some(source) = diffpack_default_loader::runtime_helpers::helper_source(id) {
            return Some(source);
        }
        self.virtual_modules.get(id).map(String::as_str)
    }

    /// Applies the opted-in Vite compile-time rewrites (`import.meta.glob`, then
    /// `import.meta.env`, then `define`, then dead-branch elimination) to a
    /// module's source before it is transformed, returning the source unchanged
    /// when the features are off or the module uses none of them. One choke point for both the serial
    /// ([`Bundler::load_module`]) and parallel ([`load_uncached`]) paths.
    ///
    /// Order matters: the two substitutions turn `process.env.NODE_ENV === 'production'`
    /// into a comparison of literals, and only then can
    /// [`crate::dead_branch`] resolve it and delete the branch that cannot run.
    /// Running here — before the module is parsed for dependencies — is what makes
    /// the dead branch's `require(...)` disappear from the graph entirely instead
    /// of being bundled but never executed.
    fn apply_vite_replacements<'s>(
        &self,
        path: &Path,
        source: &'s str,
        target: Target,
    ) -> Result<Cow<'s, str>, String> {
        let mut current = Cow::Borrowed(source);
        // `import()` with a variable in the specifier, expanded to a request -> module
        // map exactly as webpack (context module) and Rollup/Vite (dynamic import
        // vars) do. NOT opt-in: both major toolchains implement it, and a relative
        // specifier computed at runtime cannot possibly resolve from the output
        // directory. See [`crate::dynamic_import_context`].
        if let Some(rewritten) =
            diffpack_default_loader::dynamic_import_context::transform(path, &current)
        {
            current = Cow::Owned(rewritten);
        }
        if let Some(rewritten) = self.source_policy.transform(path, &current, target)? {
            current = Cow::Owned(rewritten);
        }
        Ok(current)
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
            providers: self.providers.clone(),
            provider_environment: self.provider_environment.clone(),
            provider_root: Arc::clone(&self.provider_root),
            directory: importer_directory.to_path_buf(),
            specifiers: std::array::from_fn(|_| Mutex::new(HashMap::new())),
            common_js_specifiers: std::array::from_fn(|_| Mutex::new(HashMap::new())),
            aliases: Arc::clone(&self.aliases),
            virtual_modules: Arc::clone(&self.virtual_modules),
            root: self.css_preprocess.root.as_deref().map(Arc::from),
            browser_field: Arc::clone(&self.browser_field),
        });
        shard.insert(importer_directory.to_path_buf(), Arc::clone(&cache));
        cache
    }
}

impl DirectoryResolutionCache {
    /// Resolves `specifier` as reached by `syntax` from `importer`.
    ///
    /// `resolvers` carries one resolver per syntax; the cache is per-syntax for
    /// the same reason, since the identical specifier from the identical
    /// directory can legitimately resolve to two different files.
    fn resolve(
        &self,
        resolvers: &Resolvers,
        importer: &Path,
        specifier: &str,
        syntax: ImportSyntax,
    ) -> Result<ResolvedModule, String> {
        let resolver = resolvers.for_syntax(syntax);
        let hash = hash_value(specifier);
        let shard = match syntax {
            ImportSyntax::Esm => &self.specifiers[hash as usize % self.specifiers.len()],
            ImportSyntax::CommonJs => {
                &self.common_js_specifiers[hash as usize % self.common_js_specifiers.len()]
            }
        };
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
        // A build-generated virtual module (e.g. `tanstack-start-manifest:v`)
        // resolves to itself: its id is the specifier, and the loader synthesizes
        // it from the recorded source instead of touching the filesystem.
        // A runtime helper the transform emitted an import for resolves to itself the
        // same way. Claimed BEFORE the filesystem resolver even for a helper this
        // build does not carry: the specifier names a diffpack-internal package, so
        // "cannot find module, try npm install" would be advice that cannot work.
        if diffpack_default_loader::runtime_helpers::helper_name(specifier).is_some() {
            let result =
                if diffpack_default_loader::runtime_helpers::helper_source(specifier).is_some() {
                    Ok(ResolvedModule {
                        id: SharedModuleId::from(specifier),
                        side_effect_free: true,
                        provider_external: false,
                    })
                } else {
                    Err(diffpack_default_loader::runtime_helpers::unknown_helper_error(specifier))
                };
            shard
                .lock()
                .expect("resolution specifier cache poisoned")
                .insert(specifier.to_owned(), result.clone());
            return result;
        }
        if self.virtual_modules.contains_key(specifier) {
            let result = Ok(ResolvedModule {
                id: SharedModuleId::from(specifier),
                side_effect_free: false,
                provider_external: false,
            });
            shard
                .lock()
                .expect("resolution specifier cache poisoned")
                .insert(specifier.to_owned(), result.clone());
            return result;
        }
        let provider_result = self
            .providers
            .resolve(ResolveRequest {
                specifier,
                importer: Some(importer.to_string_lossy().as_ref()),
                context: HookContext {
                    environment: &self.provider_environment,
                    project_root: &self.provider_root,
                },
            })
            .map_err(|diagnostic| {
                diagnostic
                    .provider
                    .map_or(diagnostic.message.clone(), |provider| {
                        format!("provider {provider}: {}", diagnostic.message)
                    })
            })?;
        match provider_result {
            ResolveResult::Resolved(id) => {
                let result = Ok(ResolvedModule {
                    id: id.into(),
                    side_effect_free: false,
                    provider_external: false,
                });
                shard
                    .lock()
                    .expect("resolution specifier cache poisoned")
                    .insert(specifier.to_owned(), result.clone());
                return result;
            }
            ResolveResult::External(id) => {
                let result = Ok(ResolvedModule {
                    id: id.into(),
                    side_effect_free: false,
                    provider_external: true,
                });
                shard
                    .lock()
                    .expect("resolution specifier cache poisoned")
                    .insert(specifier.to_owned(), result.clone());
                return result;
            }
            ResolveResult::NoMatch => {}
        }
        let resource = ResourceId::parse(specifier);
        let path_specifier = resource.path.as_str();
        // Aliases win before the standards-aware resolver (which would route a
        // `#`-specifier through package `imports` and fail). Two shapes:
        // an exact match on a real FILE returns it directly (the TanStack
        // virtual-entry style), while Vite's `resolve.alias` semantics also
        // rewrite PREFIX matches (`@/components/x` with `@ -> <root>/src`
        // becomes `<root>/src/components/x`) — the rewritten path then goes
        // through the normal resolver so extensions and index files apply.
        let mut aliased_specifier: Option<String> = None;
        for (from, target) in self.aliases.iter() {
            if from.as_str() == path_specifier {
                if target.is_file() {
                    let result = Ok(ResolvedModule {
                        id: module_id_with_resource(target, &resource),
                        side_effect_free: false,
                        provider_external: false,
                    });
                    shard
                        .lock()
                        .expect("resolution specifier cache poisoned")
                        .insert(specifier.to_owned(), result.clone());
                    return result;
                }
                aliased_specifier = Some(target.to_string_lossy().into_owned());
                break;
            }
            if let Some(rest) = path_specifier
                .strip_prefix(from.as_str())
                .and_then(|rest| rest.strip_prefix('/'))
            {
                aliased_specifier = Some(target.join(rest).to_string_lossy().into_owned());
                break;
            }
        }
        let original_specifier = path_specifier;
        let path_specifier = aliased_specifier.as_deref().unwrap_or(path_specifier);
        // A root-absolute specifier is a project path, not a filesystem path.
        if let Some(resolved) = self.resolve_root_absolute(path_specifier, &resource) {
            let result = Ok(ResolvedModule {
                id: resolved,
                side_effect_free: false,
                provider_external: false,
            });
            shard
                .lock()
                .expect("resolution specifier cache poisoned")
                .insert(specifier.to_owned(), result.clone());
            return result;
        }
        // Most module graphs overwhelmingly use explicit relative files. Avoid
        // the general Node resolver on a cache miss when that exact file exists;
        // all ambiguous cases still take the standards-aware path.
        //
        // A file an object-form `browser` field can rewrite is such an ambiguous
        // case: the package means for a browser build to get the REPLACEMENT, and
        // joining the path would hand back the very module the field replaces
        // (axios's `./lib/adapters/http.js`, which imports `http`/`https`/`zlib`).
        let exact_relative = path_specifier.strip_prefix("./").and_then(|relative| {
            let candidate = self.directory.join(relative);
            if !candidate.is_file() {
                return None;
            }
            let directory = candidate.parent()?;
            if self.browser_field.remaps_directory(directory) {
                return None;
            }
            Some(module_id_with_resource(&candidate, &resource))
        });
        let result = if let Some(resolved) = exact_relative {
            Ok(ResolvedModule {
                id: resolved,
                side_effect_free: false,
                provider_external: false,
            })
        } else {
            resolver
                .resolve_file(importer, path_specifier)
                .map(|resolution| {
                    // A `.node` addon resolves like any other file; that it cannot
                    // be BUNDLED is a loader concern, reported by
                    // [`unhandled_source`]. Failing it here instead turned a found
                    // file into `cannot resolve ...: install it: npm install ...`.
                    let resolved = resolution.full_path();
                    let side_effect_free = resolution.package_json().is_some_and(|package| {
                        matches!(package.side_effects(), Some(SideEffects::Bool(false)))
                    });
                    ResolvedModule {
                        id: module_id_with_resource(&resolved, &resource),
                        side_effect_free,
                        provider_external: false,
                    }
                })
                .or_else(|error| match error {
                    // `"browser": { "./lib/node.js": false }` — the package
                    // declares this module ABSENT in a browser, which the resolver
                    // reports as an ignored path. webpack and Vite both substitute
                    // an empty module here; treating it as "file not found" would
                    // fail a build the package explicitly supports.
                    ResolveError::Ignored(_) => Ok(ResolvedModule {
                        id: SharedModuleId::from(BROWSER_EXCLUDED_MODULE_ID),
                        side_effect_free: true,
                        provider_external: false,
                    }),
                    other => Err(other.to_string()),
                })
        };
        // An alias whose target is a package DIRECTORY cannot answer a SUBPATH by
        // path join. Vite's `resolve.dedupe` pins `svelte` to
        // `<root>/node_modules/svelte`, but `svelte/internal/client` is a key in
        // that package's `exports` map, not a file at that path — the join
        // produces a path that does not exist and the build fails on a package
        // that is installed. Retry the specifier AS WRITTEN from the project
        // root, which is what `dedupe` actually means (one copy, resolved from
        // the root) and lets the package's own `exports` decide. Only ever runs
        // where the build would otherwise have failed.
        let result = match result {
            Err(error) if aliased_specifier.is_some() => self
                .resolve_from_root(resolver, original_specifier, &resource)
                .ok_or(error),
            other => other,
        };
        shard
            .lock()
            .expect("resolution specifier cache poisoned")
            .insert(specifier.to_owned(), result.clone());
        result
    }

    /// Resolves `specifier` as if it were imported from a file directly in the
    /// project root, so `node_modules` lookup starts at the root and the target
    /// package's `exports` map applies. `None` when the build has no root or the
    /// specifier does not resolve there either.
    fn resolve_from_root(
        &self,
        resolver: &Resolver,
        specifier: &str,
        resource: &ResourceId,
    ) -> Option<ResolvedModule> {
        let root = self.root.as_deref()?;
        let resolution = resolver
            .resolve_file(root.join("__diffpack_root_importer__.js"), specifier)
            .ok()?;
        let side_effect_free = resolution.package_json().is_some_and(|package| {
            matches!(package.side_effects(), Some(SideEffects::Bool(false)))
        });
        Some(ResolvedModule {
            id: module_id_with_resource(&resolution.full_path(), resource),
            side_effect_free,
            provider_external: false,
        })
    }

    /// Vite's root-absolute specifier: `import icons from "/icons.svg"` means
    /// `<root>/icons.svg`, NOT the filesystem path `/icons.svg`. When no such
    /// file exists in the root but one exists under `<root>/public/`, the import
    /// is a PUBLIC file: it is copied to the site root verbatim, so it is never
    /// hashed or emitted and the module is just its URL
    /// ([`LoaderKind::PublicUrl`]).
    ///
    /// Returns `None` for every other specifier, including a genuine absolute
    /// filesystem path (a module id diffpack itself minted) — nothing under the
    /// root will match `<root>/Users/...`, so those fall through to the ordinary
    /// resolver untouched.
    fn resolve_root_absolute(
        &self,
        path_specifier: &str,
        resource: &ResourceId,
    ) -> Option<SharedModuleId> {
        let root = self.root.as_deref()?;
        let relative = path_specifier
            .strip_prefix('/')
            .filter(|rest| !rest.is_empty())?;
        let in_root = root.join(relative);
        if in_root.is_file() {
            return Some(module_id_with_resource(&in_root, resource));
        }
        let in_public = root.join("public").join(relative);
        if in_public.is_file() {
            // A query the app wrote (`/icons.svg?raw`) asks for a specific loader
            // and keeps it: the file is then read like any other source. Only the
            // plain form becomes the public URL.
            if resource.query.is_some() {
                return Some(module_id_with_resource(&in_public, resource));
            }
            let public_id = ResourceId {
                path: in_public.to_string_lossy().into_owned(),
                query: Some(LoaderKind::PublicUrl.token().to_string()),
                fragment: resource.fragment.clone(),
            };
            return Some(SharedModuleId::from(public_id.to_id()));
        }
        None
    }
}

fn hash_value(value: impl Hash) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// The module system the emitted JavaScript targets.
///
/// - [`ModuleFormat::Cjs`] renders the shared registry runtime as a
///   CommonJS-shaped IIFE (`module.exports=(()=>{…})()`, cross-chunk loading via
///   `require`); it runs under `node bundle.js` as CommonJS.
/// - [`ModuleFormat::Esm`] (the Node `server/` build) renders genuinely
///   executable ES modules (`export default`, real dynamic `import()` of split
///   chunks, `createRequire(import.meta.url)` for external Node built-ins) so
///   each emitted `.mjs` runs under Node's ESM goal, not merely passing
///   `node --check`.
/// - [`ModuleFormat::BrowserEsm`] (the client `public/` build) is the same
///   registry runtime and `export default` as [`ModuleFormat::Esm`] with real
///   dynamic `import()`, but with NO `node:module`/`createRequire` import — a
///   browser cannot resolve `node:module`. If dead server code that leaked into
///   the client graph still references a Node built-in external, `requireNative`
///   is bound to a load-safe throw-on-USE stub: property reads and construction
///   succeed (so the module LOADS and hydration proceeds), but any actual CALL
///   into the built-in throws a clear, specifically-named error — it never
///   fabricates a value.
pub use diffpack_core::{EmitOptions, ModuleFormat};

/// A minimal browser globals shim prepended to the browser-ESM entry chunk.
///
/// A browser has no Node `process` global, but React and TanStack (live client
/// code, not the leaked dead server code) read `process.env.NODE_ENV` to pick
/// their production paths. Every browser bundler defines this — Vite/webpack
/// replace `process.env.NODE_ENV` with a literal; Diffpack provides the real
/// production value as a runtime global instead. It is idempotent and never
/// clobbers a `process` the host already supplies, so it is safe to run first in
/// the entry (before any module code, and before any dynamically-imported chunk
/// loads). This is a correct value for the production client build, not a
/// fabricated stub.
/// DEV-ONLY: where a changed module lives in the current emit, for pushing a
/// targeted HMR update. See [`Bundler::hmr_locate`].
pub use diffpack_core::runtime::HmrLocation;

/// One emitted non-main chunk in the partition [`Bundler::chunk_plan`] computes.
///
/// The plan's chunks are DISJOINT and, together with the main chunk, cover every
/// live module exactly once. That is what makes the output a partition rather than
/// a pile of overlapping closures: before this, each dynamic root emitted its
/// entire static closure, so anything two routes shared (React, the router core)
/// was duplicated into every chunk that reached it.
pub use diffpack_core::{VisualizationEdge, VisualizationGraph, VisualizationNode};

/// The whole-graph derivation every emit performs before it renders a single chunk:
/// which modules survive export-level dead-module elimination, the dense order that
/// fixes every runtime id, and the chunk partition each module lands in.
///
/// Deriving it walks the entire graph — on cal.com's 18 MB client graph,
/// `live_modules` is ~12 ms and `chunk_plan` ~25 ms — and every byte a later HMR
/// micro-chunk emits has to agree with it. Both facts point the same way: derive it
/// once, in the emit that produced the bundle the browser and the dev server's Node
/// processes are actually running, and reuse it verbatim. See `Bundler::emit_plan`.
pub struct Bundler {
    #[doc(hidden)]
    pub graph: ModuleGraph<ModuleState>,
    resolver: Resolvers,
    resolution_cache: ResolutionCache,
    frontend_pool: ThreadPool,
    compiler: Arc<dyn ModuleCompiler>,
    special_modules: Arc<dyn SpecialModulePolicy>,
    runtime_policy: Arc<dyn RuntimeIntegrationPolicy>,
    output_policy: Arc<dyn OutputIntegrationPolicy>,
    target: Target,
    /// DEV-ONLY Fast Refresh / `import.meta.hot` instrumentation flag (mirrors
    /// [`BuildConfig::hmr`]). Always `false` for `build-app`.
    hmr: bool,
    /// Per-chunk render cache (interior-mutable so emit stays `&self`). Persists
    /// across incremental re-emits within one bundler, so a leaf edit re-renders
    /// only the chunk that changed and reuses every other chunk's bytes.
    render_cache: Mutex<RenderCache>,
    /// The whole-graph plan each entry chunk's most recent emit derived, keyed by
    /// entry chunk file name (interior-mutable so emit stays `&self`, like
    /// `render_cache`). Read by the dev server's HMR micro-chunk path, which must
    /// agree with the emitted bundle rather than with a freshly re-derived one.
    /// See [`EmitPlan`].
    emit_plans: Mutex<HashMap<String, Arc<EmitPlan>>>,
    /// The build configuration this bundler was constructed with, kept so a
    /// module-worker entry can be bundled with the SAME config as a nested,
    /// self-contained build at emit time.
    config: BuildConfig,
    /// The directory every emitted source-map `sources` label is relative to,
    /// resolved once per build (see [`Self::map_source_root`]). It must be the same
    /// answer for every chunk — a `sources` URL is a module's identity across the
    /// whole build — so it is computed once and cached, never per map.
    map_root: OnceLock<Option<PathBuf>>,
    /// The Tailwind class candidates last scanned, keyed by the inputs that produce
    /// them: the scan root, the output root it excludes, and the entry sheet's own
    /// text (which carries the `@source` include/exclude set). Interior-mutable so
    /// emit stays `&self`.
    ///
    /// The scan reads every source file under the root — on a monorepo app that is the
    /// dominant cost of compiling the sheet — and its result depends on those files,
    /// NOT on this graph. So it stays valid exactly as long as those files do, which is
    /// a fact only the caller knows: the dev loop calls
    /// [`Self::refresh_tailwind_scan_path`] for each file it rebuilds (re-tokenizing
    /// just that file) and [`Self::invalidate_tailwind_scan`] when the module set
    /// itself changed, so files may have appeared or vanished. Nothing invalidates it
    /// implicitly; a stale entry would compile the wrong utilities.
    tailwind_scan_cache: Mutex<HashMap<(PathBuf, PathBuf, String), TailwindScan>>,
    /// The last few COMPILED Tailwind sheets, keyed by everything the compile is a
    /// function of: the entry text, the candidate set and the app's theme. A
    /// compile is pure in those inputs, and for a sheet that delegates to the app's
    /// own Tailwind it costs a Node process (~250 ms on cal.com), so repeating it
    /// for the same inputs is pure waste — which is what the dev loop's deferred
    /// chunk compaction used to do to every edit's sheet, one full delegate per
    /// compaction pass.
    ///
    /// Small and LRU-by-insertion (a dev session edits one sheet over and over, so
    /// only the newest entries can hit); a compiled sheet is hundreds of KB, so this
    /// is deliberately not allowed to grow with the session.
    tailwind_sheet_cache: Mutex<Vec<(TailwindSheetKey, Arc<String>)>>,
}

/// A Tailwind candidate scan, kept PER FILE so an edit costs one file read.
///
/// The scan is what makes compiling a monorepo's stylesheet expensive: it reads and
/// tokenizes every source file under the root (~660 ms on cal.com). Dropping all of it
/// whenever any source changed meant paying that again after every keystroke, on the
/// thread that answers file events.
///
/// Only the per-file halves are cached. The candidate set is NOT a union of them — an
/// identifier referenced in one file resolves against a binding declared in another —
/// so it is recomputed by re-running the cross-file resolve
/// ([`diffpack_default_loader::tailwind::resolve_scans`]) over the cached parts, which reads no files and
/// tokenizes nothing. Same algorithm as a from-scratch scan, same bytes out.
struct TailwindScan {
    per_file: HashMap<PathBuf, diffpack_default_loader::tailwind::SourceScan>,
}

impl TailwindScan {
    fn candidates(&self) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        diffpack_default_loader::tailwind::resolve_scans(self.per_file.values(), &mut out);
        out
    }
}

/// What a compiled Tailwind sheet is a function of. The candidate set and theme are
/// hashed rather than stored: they are large, and a mismatch only needs to be
/// DETECTED, never explained.
#[derive(PartialEq, Eq, Clone)]
struct TailwindSheetKey {
    css: String,
    candidates: u64,
    theme: u64,
}

impl Bundler {
    pub fn discover_with_driver_policies(
        entry: &Path,
        config: &BuildConfig,
        providers: ProviderPipeline,
        policies: DriverPolicies,
    ) -> Result<(Self, BuildUpdate), String> {
        Self::discover_inner(
            entry,
            config,
            providers,
            policies.compiler,
            policies.special_modules,
            policies.runtime,
            policies.output,
            policies.source,
        )
    }

    fn discover_inner(
        entry: &Path,
        config: &BuildConfig,
        providers: ProviderPipeline,
        compiler: Arc<dyn ModuleCompiler>,
        special_modules: Arc<dyn SpecialModulePolicy>,
        runtime_policy: Arc<dyn RuntimeIntegrationPolicy>,
        output_policy: Arc<dyn OutputIntegrationPolicy>,
        source_policy: Arc<dyn SourceIntegrationPolicy>,
    ) -> Result<(Self, BuildUpdate), String> {
        let entry_path = entry
            .canonicalize()
            .map_err(|error| format!("cannot open entry {}: {error}", entry.display()))?;
        let entry_id = module_id(&entry_path);
        let resolver = Resolvers::new(&ResolverConfig {
            conditions: config.conditions.clone(),
            main_fields: config.main_fields.clone(),
            browser: config.target == Target::Client,
        });
        // Use every core: parse/transform dominates cold-build CPU and scales
        // near-linearly (each module is independent). The old `.min(4)` cap
        // held a 32-core machine to ~2.7 CPUs utilized on a 1000-module cold
        // build — the single largest cold-wall-time cost found by profiling.
        let frontend_threads = std::thread::available_parallelism().map_or(1, usize::from);
        let mut bundler = Self {
            graph: ModuleGraph::new(),
            resolver,
            compiler,
            special_modules,
            runtime_policy,
            output_policy,
            resolution_cache: ResolutionCache::new(
                providers,
                config
                    .aliases
                    .iter()
                    .map(|(from, to)| (from.clone(), PathBuf::from(to)))
                    .collect(),
                config.virtual_modules.clone(),
                source_policy,
                &config.base,
                entry_path
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| PathBuf::from(".")),
                config.asset_inline_limit,
                config.scss.clone(),
                config.image_import_shape,
                config.css_preprocess.clone(),
                config.jsx_extensions,
                config.jsx.clone(),
                // Same condition as `resolve_options`' `alias_fields`: only a
                // browser build applies the `browser` field.
                config.target == Target::Client,
                if config.hmr {
                    BuildMode::Development
                } else {
                    BuildMode::Production
                },
                config.server_external_packages.clone(),
            ),
            frontend_pool: ThreadPoolBuilder::new()
                .num_threads(frontend_threads)
                .thread_name(|index| format!("diffpack-frontend-{index}"))
                .build()
                .map_err(|error| format!("cannot create frontend worker pool: {error}"))?,
            target: config.target,
            hmr: config.hmr,
            config: config.clone(),
            render_cache: Mutex::new(RenderCache::default()),
            emit_plans: Mutex::new(HashMap::new()),
            map_root: OnceLock::new(),
            tailwind_scan_cache: Mutex::new(HashMap::new()),
            tailwind_sheet_cache: Mutex::new(Vec::new()),
        };
        bundler.graph.entry = bundler.graph.intern(entry_id.clone());

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

    /// Add independently-emittable roots to this configured compiler session.
    /// Modules already present in the graph are reused verbatim, so resolution,
    /// parsing, and transformation happen once per environment rather than once
    /// per entry. Each root can later be selected with [`Self::select_entry`].
    pub fn discover_additional_entries(
        &mut self,
        entries: &[PathBuf],
    ) -> Result<BuildUpdate, String> {
        let mut roots = Vec::with_capacity(entries.len());
        for entry in entries {
            let path = entry
                .canonicalize()
                .map_err(|error| format!("cannot open entry {}: {error}", entry.display()))?;
            let id = module_id(&path);
            let index = self.graph.intern(id.clone());
            if self.graph.modules[index].is_none() {
                roots.push(id);
            }
        }
        let mut delta = GraphDelta::default();
        let mut diagnostics = Vec::new();
        let transformed_modules = self.discover_from(roots, &mut delta, &mut diagnostics, false)?;
        Ok(BuildUpdate {
            delta,
            transformed_modules,
            diagnostics,
        })
    }

    /// Select one previously-discovered root as the entry for reachability,
    /// linking, and emission. Selection is cheap and does not rediscover modules.
    pub fn select_entry(&mut self, entry: &Path) -> Result<(), String> {
        let path = entry
            .canonicalize()
            .map_err(|error| format!("cannot open entry {}: {error}", entry.display()))?;
        let id = module_id(&path);
        let Some(&index) = self.graph.indices.get(id.as_ref()) else {
            return Err(format!(
                "entry {} was not discovered in this compiler session",
                entry.display()
            ));
        };
        if self.graph.modules[index].is_none() {
            return Err(format!(
                "entry {} has no compiled module in this compiler session",
                entry.display()
            ));
        }
        self.graph.entry = index;
        Ok(())
    }

    /// Whether `path` is already a loaded module in this environment's graph.
    /// A long-lived dev server uses this to distinguish an EDIT to an existing
    /// module (supported: incremental rebuild) from a NEW file appearing
    /// (unsupported by the full-page-reload slice — it needs route-tree
    /// regeneration), so the latter can hard-error instead of silently no-op'ing.
    pub fn is_known_module(&self, path: &Path) -> bool {
        let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let id = module_id(&path);
        // A physical file is "known" when it is a module itself, when a
        // query-derived virtual module reads from it (`x.css?media=screen`,
        // `logo.png?url`), or when a media-qualified CSS module inlined its
        // content — in every case [`Self::rebuild_path`] can meaningfully apply
        // an edit to it.
        self.graph
            .indices
            .get(id.as_ref())
            .and_then(|&index| self.graph.modules[index].as_ref())
            .is_some()
            || !self.derived_virtual_siblings(id.as_ref()).is_empty()
            || !self.css_inline_dependents(&path).is_empty()
    }

    pub fn rebuild_path(&mut self, path: &Path) -> Result<BuildUpdate, String> {
        let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let id = module_id(&path);
        let mut delta = GraphDelta::default();
        let mut diagnostics = Vec::new();
        let mut transformed_modules = 0;

        let known = self
            .graph
            .indices
            .get(id.as_ref())
            .copied()
            .filter(|&index| self.graph.modules[index].is_some());
        if let Some(index) = known {
            let Some(old) = self.graph.modules[index].clone() else {
                unreachable!("known index always holds a module");
            };
            if !path.is_file() {
                delta.changed.insert(id.to_string());
                for (_, target, _) in &old.dependencies {
                    delta
                        .edge_updates
                        .push(((id.to_string(), self.graph.ids[*target].to_string()), -1));
                }
                self.graph.modules[index] = None;
                return Ok(BuildUpdate {
                    delta,
                    transformed_modules: 0,
                    diagnostics,
                });
            }
            transformed_modules +=
                self.reload_known_module(index, id.as_ref(), &path, &mut delta, &mut diagnostics)?;
        }

        // Derived virtual modules read their source from this same physical file —
        // notably the route `?tsr-split=<target>` chunks, whose actual component /
        // loader bodies live in the file that just changed, and the CSS
        // `?media=<query>` modules built from it. A physical-file edit must
        // re-derive every such sibling, or the derived module on disk would keep
        // the pre-edit content. Each sibling is loaded from its full id string
        // (which carries the loader query). This runs even when the bare path is
        // not a module itself (e.g. a CSS file only ever imported with a media
        // query).
        for (sibling_index, sibling_id) in self.derived_virtual_siblings(id.as_ref()) {
            transformed_modules += self.reload_known_module(
                sibling_index,
                &sibling_id,
                Path::new(&sibling_id),
                &mut delta,
                &mut diagnostics,
            )?;
        }

        // Modules whose emitted CSS INLINED this file's content (a
        // media-qualified `@import`'s nested imports) must also re-derive, or
        // their stylesheet text would keep the pre-edit bytes. The recorded
        // `css_source_files` lists are transitively flattened at load time, so
        // one pass suffices.
        for (dependent_index, dependent_id) in self.css_inline_dependents(&path) {
            transformed_modules += self.reload_known_module(
                dependent_index,
                &dependent_id,
                Path::new(&dependent_id),
                &mut delta,
                &mut diagnostics,
            )?;
        }

        Ok(BuildUpdate {
            delta,
            transformed_modules,
            diagnostics,
        })
    }

    /// Every currently-loaded module whose emitted CSS inlined the content of
    /// the physical file at `path` (recorded in `css_source_files`). These must
    /// be re-derived when that file changes. Returns `(dense index, full id)`
    /// pairs.
    fn css_inline_dependents(&self, path: &Path) -> Vec<(DenseModuleId, String)> {
        self.graph
            .ids
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                self.graph.modules[*index]
                    .as_ref()
                    .is_some_and(|module| module.css_source_files.iter().any(|file| file == path))
            })
            .map(|(index, id)| (index, id.to_string()))
            .collect()
    }

    /// Reload one already-known module (dense `index`, string `id`, and the
    /// `load_path` the loader reads — for a query-bearing virtual module this is
    /// its full id string). Diffs the reloaded hash and dependency edges into
    /// `delta` and discovers any newly-referenced modules. Returns the count of
    /// modules (re)transformed (this one plus any newly discovered dependency).
    fn reload_known_module(
        &mut self,
        index: usize,
        id: &str,
        load_path: &Path,
        delta: &mut GraphDelta,
        diagnostics: &mut Vec<Diagnostic>,
    ) -> Result<usize, String> {
        let Some(old) = self.graph.modules[index].clone() else {
            return Ok(0);
        };
        let new = self.load_module(load_path, diagnostics)?;
        // "Changed" means the module's EMITTED output changed (so its chunk must be
        // re-rendered) — not merely that its source text moved. A route edit whose
        // body was split into another chunk leaves the reference module's output
        // byte-identical, so it is correctly not marked changed and its (large)
        // entry chunk is reused.
        if old.code_hash != new.code_hash {
            delta.changed.insert(id.to_string());
        }
        let old_edges = old
            .dependencies
            .iter()
            .map(|(_, target, _)| target)
            .map(|target| (id.to_string(), self.graph.ids[*target].to_string()))
            .collect::<BTreeSet<_>>();
        let new_edges = new
            .dependencies
            .iter()
            .map(|(_, target, _)| target)
            .map(|target| (id.to_string(), self.graph.ids[*target].to_string()))
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
            .filter(|dependency| self.graph.modules[**dependency].is_none())
            .map(|dependency| self.graph.ids[*dependency].clone())
            .collect::<Vec<_>>();
        self.graph.modules[index] = Some(new);
        Ok(1 + self.discover_from(new_paths, delta, diagnostics, true)?)
    }

    /// Every currently-loaded module whose loader id has the same filesystem path
    /// as `path_id` but carries a query or fragment — i.e. a virtual module
    /// derived from that physical file (a `?tsr-split=*` route chunk, a `?url`
    /// asset, a `?raw` inline). These must be re-derived when the physical file
    /// changes. Returns `(dense index, full id string)` pairs.
    fn derived_virtual_siblings(&self, path_id: &str) -> Vec<(DenseModuleId, String)> {
        self.graph
            .ids
            .iter()
            .enumerate()
            .filter(|(index, id)| {
                self.graph.modules[*index].is_some() && {
                    let resource = ResourceId::parse(id.as_ref());
                    (resource.query.is_some() || resource.fragment.is_some())
                        && resource.path == path_id
                }
            })
            .map(|(index, id)| (index, id.to_string()))
            .collect()
    }

    pub fn emit(&self, reachable: &BTreeSet<ModuleId>, output: &Path) -> Result<EmitStats, String> {
        self.emit_with_options(reachable, output, EmitOptions::default())
    }

    /// [`Self::emit_public_incremental`] that can be STOPPED part-way. Returns
    /// `(rendered chunks, cancelled)`.
    ///
    /// See [`EmitCancel`]. Only the dev loop's deferred compaction passes a real
    /// signal: it runs on the same thread that answers file events, and a chunk
    /// render on a large app takes hundreds of milliseconds, so without this an edit
    /// arriving mid-compaction waited it out.
    pub fn emit_public_incremental_cancellable(
        &self,
        reachable: &BTreeSet<ModuleId>,
        output_root: &Path,
        options: EmitOptions,
        cancel: EmitCancel<'_>,
    ) -> Result<(usize, bool), String> {
        let options = EmitOptions {
            format: ModuleFormat::BrowserEsm,
            ..options
        };
        let public_dir = output_root.join("public");
        let stats = self.emit_inner(reachable, &public_dir.join("client.js"), options, cancel)?;
        Ok((stats.rendered_chunks, stats.cancelled))
    }

    /// [`Self::emit_server_into`] that can be STOPPED part-way. Returns
    /// `(summary, cancelled)`, and when cancelled it does NOT prune: a partial emit's
    /// written set does not describe the tree, so pruning against it would delete
    /// live chunks.
    pub fn emit_server_into_cancellable(
        &self,
        reachable: &BTreeSet<ModuleId>,
        server_dir: &Path,
        options: EmitOptions,
        cancel: EmitCancel<'_>,
    ) -> Result<(EmitSummary, bool), String> {
        let options = EmitOptions {
            format: ModuleFormat::Esm,
            ..options
        };
        let server_dir = server_dir.to_path_buf();
        let mut stats =
            self.emit_inner(reachable, &server_dir.join("server.mjs"), options, cancel)?;
        if stats.cancelled {
            let mut summary = EmitSummary::of(&server_dir)?;
            summary.rendered_chunks = stats.rendered_chunks;
            return Ok((summary, true));
        }
        stats.written.extend(
            self.output_policy
                .write_server_runtime(&server_dir, options.hmr)?,
        );
        diffpack_default_loader::output::prune_output(&server_dir, &stats.written)?;
        let mut summary = EmitSummary::of(&server_dir)?;
        summary.rendered_chunks = stats.rendered_chunks;
        Ok((summary, false))
    }

    /// The number of chunk renders currently cached. Bounded to the live chunk
    /// set by per-emit eviction, so it stays flat across a long edit sequence;
    /// exposed for the memory guards in `docs/THESIS_GUARDS.md`.
    pub fn render_cache_len(&self) -> usize {
        self.render_cache.lock().unwrap().entries.len()
    }

    /// Drop the cached Tailwind candidate scan entirely: the module set changed, so
    /// files this graph compiles against may have been added or removed and the scan's
    /// file list is no longer the truth. A CHANGED file is handled without this, by
    /// [`Self::refresh_tailwind_scan_path`].
    ///
    /// The dev loop calls this for every non-stylesheet rebuild. It is deliberately
    /// explicit — the scan reads the file system rather than this graph, so nothing
    /// inside the bundler can observe that its inputs moved.
    pub fn invalidate_tailwind_scan(&self) {
        self.tailwind_scan_cache.lock().unwrap().clear();
    }

    /// Emits the client browser build into `<output_root>/public/`: the entry
    /// JavaScript chunk (`client.js`), its dynamic-import chunks, the extracted
    /// stylesheet, and every content-hashed asset under `public/assets/`. The
    /// `public/` directory is rebuilt from scratch so stale files never linger,
    /// and the returned [`EmitSummary`] counts exactly what landed on disk.
    ///
    /// This drives the existing single-output emit at a `public/` layout; it is a
    /// build-time entry point (off the incremental hot path), so the thesis guards
    /// are unaffected.
    pub fn emit_public(
        &self,
        reachable: &BTreeSet<ModuleId>,
        output_root: &Path,
        options: EmitOptions,
    ) -> Result<EmitSummary, String> {
        self.emit_public_preserving(reachable, output_root, options, &BTreeSet::new())
    }

    /// [`Self::emit_public`], plus paths under `public/` that this emit does NOT own
    /// and its prune must therefore leave alone.
    ///
    /// The prune deletes every file under `public/` the CLIENT graph did not just
    /// write, which is right for the client's own stale chunks and wrong for a file
    /// another graph published there. `public/rsc.css` is exactly that: the
    /// react-server graph is authoritative for the app's CSS and preserves its
    /// compiled sheet to that path, and in a production `build-app` the two graphs are
    /// separate processes, so the react-server copy simply lands after the client's
    /// prune. `diffpack dev` builds both in ONE process with the react-server graph
    /// FIRST, so the client's prune deleted the sheet it had just written: the document
    /// still linked `/rsc.css` (the link is guarded on the artifact beside the render
    /// bundle, which survives), `GET /rsc.css` 404ed, and the page rendered unstyled —
    /// on cal.com, and on `integration/next-app-router` from a cold dev boot.
    pub fn emit_public_preserving(
        &self,
        reachable: &BTreeSet<ModuleId>,
        output_root: &Path,
        options: EmitOptions,
        preserve: &BTreeSet<PathBuf>,
    ) -> Result<EmitSummary, String> {
        // The client build is always browser-executable ESM, regardless of the
        // caller's options. The SSR document injects `client.js` as
        // `<script type="module">`, so a CJS `module.exports=…` entry would throw
        // `module is not defined` on load and the app would never hydrate. Browser
        // ESM emits `export default` with real dynamic `import()` and NO
        // `node:module` import, so the entry loads and runs in the browser.
        let options = EmitOptions {
            format: ModuleFormat::BrowserEsm,
            ..options
        };
        let public_dir = output_root.join("public");
        let stats = self.emit_environment(reachable, &public_dir, "client.js", options)?;
        let mut keep = stats.written.clone();
        keep.extend(preserve.iter().cloned());
        diffpack_default_loader::output::prune_output(&public_dir, &keep)?;
        let mut summary = EmitSummary::of(&public_dir)?;
        summary.rendered_chunks = stats.rendered_chunks;
        Ok(summary)
    }

    /// The LEAN incremental client re-emit for an HMR hot update: write only the
    /// chunk(s) whose bytes actually changed and return how many were rendered.
    /// Unlike [`Self::emit_public`], it does NOT prune stale files or walk the output
    /// tree to build a full [`EmitSummary`] — a same-graph edit (the HMR fast path,
    /// which the caller has already confirmed did not change the module graph)
    /// produces no stale files, and those two full `public/` directory walks are the
    /// dominant cost of an otherwise sub-millisecond incremental chunk render.
    pub fn emit_public_incremental(
        &self,
        reachable: &BTreeSet<ModuleId>,
        output_root: &Path,
        options: EmitOptions,
    ) -> Result<usize, String> {
        let options = EmitOptions {
            format: ModuleFormat::BrowserEsm,
            ..options
        };
        let public_dir = output_root.join("public");
        let stats = self.emit_environment(reachable, &public_dir, "client.js", options)?;
        Ok(stats.rendered_chunks)
    }

    /// Compile and write ONLY this graph's stylesheet (`<entry-stem>.css` beside
    /// `output`), rendering no JS chunk and touching no manifest or asset.
    ///
    /// A stylesheet edit changes exactly one artifact the browser needs, and it can
    /// be delivered by swapping one `<link>` — but the sheet is produced as a
    /// side-product of the full environment emit, which on a real app costs ~1.2 s
    /// of chunk rendering that a css edit does not need. That coupling is why a css
    /// edit used to wait for the deferred compaction pass: it was the next time
    /// anything wrote the sheet. This is the same stylesheet pipeline
    /// (`Self::emit_css`: candidate scan, Tailwind compile, concatenation in
    /// static execution order) with the chunk half left out, so the dev loop can run
    /// it on the edit itself.
    ///
    /// Returns the path written, or `None` when this graph compiles no CSS at all.
    /// The bytes are written only if they differ, so an edit that does not move the
    /// stylesheet leaves its mtime — and any conditional-request cache — alone.
    pub fn emit_stylesheet_only(
        &self,
        reachable: &BTreeSet<ModuleId>,
        output: &Path,
        cancel: EmitCancel<'_>,
    ) -> Result<StylesheetEmit, String> {
        let parent = output
            .parent()
            .ok_or_else(|| format!("output has no parent: {}", output.display()))?;
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
        // The same live set the full emit derives, so the sheet contains exactly the
        // modules the emitted bundle would have contributed — no more, no less.
        let reachable = self.live_modules(reachable);
        let allowed = reachable
            .iter()
            .filter_map(|id| self.graph.indices.get(id.as_str()).copied())
            .collect::<HashSet<_>>();
        let mut written = BTreeSet::new();
        if self.emit_css(&allowed, output, &mut written, cancel)? {
            return Ok(StylesheetEmit::Cancelled);
        }
        match written.into_iter().next() {
            Some(sheet) => Ok(StylesheetEmit::Written(sheet)),
            None => Ok(StylesheetEmit::NoStylesheet),
        }
    }

    /// Emits a generic web build directly into `output_dir`: the browser-ESM
    /// entry chunk (named `entry_file`, e.g. `index.js`), its dynamic-import
    /// chunks, the extracted stylesheet beside the entry, and content-hashed
    /// assets under `assets/`. Mirrors [`Self::emit_public`] but without the
    /// TanStack `public/` nesting: `output_dir` IS the site root (the caller
    /// writes `index.html` and any static files after this, since the stale-file
    /// prune here only keeps what the emit itself wrote).
    pub fn emit_web(
        &self,
        reachable: &BTreeSet<ModuleId>,
        output_dir: &Path,
        entry_file: &str,
        options: EmitOptions,
    ) -> Result<EmitSummary, String> {
        let (summary, written) =
            self.emit_web_written(reachable, output_dir, entry_file, options)?;
        diffpack_default_loader::output::prune_output(output_dir, &written)?;
        Ok(summary)
    }

    /// Emit one browser page into `output_dir` (entry chunk `entry_file`, its
    /// dynamic-import chunks, extracted `<entry-stem>.css`, and content-hashed
    /// assets) WITHOUT pruning stale files, returning the summary and the exact
    /// set of files written. This is the multi-page primitive: a MULTI-PAGE build
    /// emits every page into a shared `output_dir` (page chunks named per page,
    /// assets deduped by content hash), accumulates every page's `written` set, and
    /// prunes ONCE at the end via [`diffpack_default_loader::output::prune_output`] — so a shared asset written
    /// by page A is never deleted by page B's emit, while stale files from a prior
    /// build are still removed. A single-page build ([`Self::emit_web`]) prunes
    /// immediately against the one page's set, unchanged.
    pub fn emit_web_written(
        &self,
        reachable: &BTreeSet<ModuleId>,
        output_dir: &Path,
        entry_file: &str,
        options: EmitOptions,
    ) -> Result<(EmitSummary, BTreeSet<PathBuf>), String> {
        let options = EmitOptions {
            format: ModuleFormat::BrowserEsm,
            ..options
        };
        let stats = self.emit_environment(reachable, output_dir, entry_file, options)?;
        let mut summary = EmitSummary::of(output_dir)?;
        summary.rendered_chunks = stats.rendered_chunks;
        Ok((summary, stats.written))
    }

    /// Emits the server (SSR) build into `<output_root>/server/` as Node ESM
    /// `.mjs` modules, mirroring [`Self::emit_public`]: the entry chunk
    /// (`server/server.mjs`), its dynamic-import chunks
    /// (`server/server.chunk-N.mjs`), the extracted stylesheet, and every
    /// content-hashed asset under `server/assets/`. The `server/` directory is
    /// rebuilt from scratch so stale files never linger.
    ///
    /// The output uses the `.mjs` extension so Node treats each chunk as an ES
    /// module. This is the foundation slice: it produces the server module graph
    /// but not yet the Node HTTP runtime entry (`server/index.mjs`) nor the
    /// natively-generated TanStack manifests — those are the next slices and are
    /// deliberately not faked here.
    pub fn emit_server(
        &self,
        reachable: &BTreeSet<ModuleId>,
        output_root: &Path,
        options: EmitOptions,
    ) -> Result<EmitSummary, String> {
        self.emit_server_into(reachable, &output_root.join("server"), options)
    }

    /// [`Self::emit_server`] with the server directory itself named by the
    /// caller. The production orchestrator points the react-server graph straight
    /// at `<out>/rsc-render/` so it can build concurrently with the ssr graph
    /// (which owns `<out>/server/`) instead of emitting into `server/` and being
    /// copied aside before ssr may start.
    pub fn emit_server_into(
        &self,
        reachable: &BTreeSet<ModuleId>,
        server_dir: &Path,
        options: EmitOptions,
    ) -> Result<EmitSummary, String> {
        // The server build is always Node ESM, regardless of the caller's
        // options, so every emitted `.mjs` executes under Node's ESM goal.
        let options = EmitOptions {
            format: ModuleFormat::Esm,
            ..options
        };
        let server_dir = server_dir.to_path_buf();
        let mut stats = self.emit_environment(reachable, &server_dir, "server.mjs", options)?;
        // Emit the Node HTTP runtime entry (`server/index.mjs`) and its sibling
        // SSR/router runtime modules on top of the module graph. Their paths join
        // the kept set so the stale-file prune never deletes them, and `EmitSummary`
        // is recomputed afterwards so it counts the runtime files too.
        stats.written.extend(
            self.output_policy
                .write_server_runtime(&server_dir, options.hmr)?,
        );
        diffpack_default_loader::output::prune_output(&server_dir, &stats.written)?;
        let mut summary = EmitSummary::of(&server_dir)?;
        summary.rendered_chunks = stats.rendered_chunks;
        Ok(summary)
    }

    /// Emits the environment's entry chunk (named `entry_file`, whose extension —
    /// `.js` or `.mjs` — flows onto every dynamic-import chunk) plus its CSS and
    /// assets, and returns the [`EmitStats`] describing what was re-rendered and
    /// which files are kept. Unlike a from-scratch rebuild, this does NOT wipe the
    /// output tree: [`Self::emit_with_options`] writes only the chunks whose bytes
    /// changed, and the caller prunes files no longer in `stats.written`, so an
    /// incremental re-emit touches only the chunk that changed while preserving the
    /// "no stale files linger" guarantee.
    fn emit_environment(
        &self,
        reachable: &BTreeSet<ModuleId>,
        output_dir: &Path,
        entry_file: &str,
        options: EmitOptions,
    ) -> Result<EmitStats, String> {
        fs::create_dir_all(output_dir)
            .map_err(|error| format!("cannot create {}: {error}", output_dir.display()))?;
        let entry_output = output_dir.join(entry_file);
        self.emit_with_options(reachable, &entry_output, options)
    }

    pub fn emit_with_options(
        &self,
        reachable: &BTreeSet<ModuleId>,
        output: &Path,
        options: EmitOptions,
    ) -> Result<EmitStats, String> {
        self.emit_inner(reachable, output, options, EmitCancel::never())
    }

    /// [`Self::emit_with_options`], plus the [`EmitCancel`] the dev loop's deferred
    /// compaction hands in. Checked before each expensive phase and inside the
    /// per-module render fan-out; on cancellation nothing further is written and
    /// `stats.cancelled` says so.
    fn emit_inner(
        &self,
        reachable: &BTreeSet<ModuleId>,
        output: &Path,
        options: EmitOptions,
        cancel: EmitCancel<'_>,
    ) -> Result<EmitStats, String> {
        // A truthful source map can only come from the per-module maps the
        // TRANSFORM produced, and those are produced only when the bundler was
        // built with `BuildConfig::source_maps`. Emitting maps from a bundler that
        // was not is impossible to do honestly, so it is refused here — loudly and
        // with the fix named — instead of falling back to positions nobody
        // measured.
        if options.source_map && !self.config.source_maps {
            return Err(format!(
                "cannot write source maps for {}: this bundler was built without \
                 `BuildConfig::source_maps`, so no module carries the Oxc printer's real \
                 positions. Set `source_maps: true` in the BuildConfig used to discover the \
                 graph (the CLI does this when `--sourcemap` is passed).",
                output.display()
            ));
        }
        let parent = output
            .parent()
            .ok_or_else(|| format!("output has no parent: {}", output.display()))?;
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
        let mut stats = EmitStats::default();
        // Set by any check point that saw the cancel signal, including from inside
        // the rayon fan-out below.
        let cancelled = std::sync::atomic::AtomicBool::new(false);
        if cancel.cancelled() {
            stats.cancelled = true;
            return Ok(stats);
        }
        // Keys of every chunk this emit renders or reuses; entries not among them
        // are evicted at the end so the cache stays bounded to the live chunk set.
        let mut live_keys = HashSet::new();
        // Generic, `sideEffects`-aware dead-module elimination: refine the
        // module-level reachable set down to the export-level LIVE set before
        // emit, so a reachable-but-unused `sideEffects:false` module (and its
        // now-orphaned `node:` requires) never reaches the output. Deterministic,
        // so incremental and full builds emit byte-identical bytes.
        let live_stage = diffpack_core::build_profile::stage("emit/live-modules");
        let reachable = self.live_modules(reachable);
        drop(live_stage);
        let reachable_dense = reachable
            .iter()
            .filter_map(|id| self.graph.indices.get(id.as_str()).copied())
            .collect::<Vec<_>>();
        let allowed = reachable_dense.iter().copied().collect::<HashSet<_>>();
        if cancel.cancelled() {
            stats.cancelled = true;
            return Ok(stats);
        }
        let assets_stage = diffpack_core::build_profile::stage("emit/assets");
        self.emit_assets(&allowed, parent, &mut stats.written, cancel)?;
        drop(assets_stage);
        let mut runtime_ids = vec![None; self.graph.ids.len()];
        for (runtime_id, &dense_id) in reachable_dense.iter().enumerate() {
            runtime_ids[dense_id] = Some(runtime_id);
        }
        let main_modules = self.static_closure(self.graph.entry, &allowed);
        // Export demand is aggregated once over EVERY reachable module (not per
        // chunk), so a module keeps the exports any chunk imports from it even
        // when the consumer lands in a different chunk than the definition.
        let global_demands = self.export_demands(&reachable_dense);
        let entry_name = output
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| format!("output path is not UTF-8: {}", output.display()))?;
        // The partition of everything outside the entry's static closure. Chunk
        // membership, file names and load order all come from here, so the files
        // on disk and the `import()` references rewritten into them agree by
        // construction.
        if cancel.cancelled() {
            stats.cancelled = true;
            return Ok(stats);
        }
        let plan_stage = diffpack_core::build_profile::stage("emit/chunk-plan");
        let plans = self.chunk_plan(&allowed, entry_name)?;
        drop(plan_stage);
        // Publish what this emit derived. A later HMR micro-chunk for this entry
        // reuses it verbatim rather than re-deriving from a graph that has since
        // moved on, which is both ~37 ms cheaper per micro-chunk and the only way
        // its runtime ids can match the bundle these bytes are about to become.
        self.record_emit_plan(
            entry_name,
            Arc::new(self.build_emit_plan(reachable_dense.clone(), allowed.clone(), &plans)),
        )?;
        let chunk_names = chunk_names(&plans);
        // Every chunk's file name, which is also its chunk id: the RSC seam's id -> URL
        // table must cover shared chunks too, and `chunk_names` only knows the ones that
        // own a dynamic-import root (see `EmitPlan::chunk_files`).
        let chunk_files: Vec<String> = plans.iter().map(|plan| plan.file_name.clone()).collect();
        // The scope-hoisted flat render concatenates a chunk's modules into one
        // scope, which is only sound when the chunk carries every module its
        // members statically reference. Splitting shared code out breaks that for
        // any chunk with a cross-chunk edge, and a flat chunk cannot be mixed with
        // a registry chunk either: the two speak different protocols. A flat chunk
        // publishes bindings (`export{a,b}` / `module.exports=`), while
        // `require.dynamic` resolves a registry id through `__require`, so a flat
        // chunk consumed by a registry main chunk registers no factory and fails
        // with "Module is not loaded". Per-chunk eligibility cannot express that,
        // because the MAIN chunk independently falls back to the registry whenever
        // flat rendering bails (an external binding, a duplicate top-level
        // declaration). So the protocol is decided for the whole build by whether
        // any split chunk exists at all: a single-chunk build keeps scope hoisting,
        // and any split build uses the registry everywhere.
        //
        // This forgoes scope hoisting for split builds. Recovering it needs
        // cross-chunk binding imports (what Rollup emits), which is a real feature,
        // not a tweak of this flag.
        let flat_allowed = plans.is_empty();
        // Module workers: each `(key, entry)` a live module declared becomes a
        // nested, self-contained bundle under `assets/`; the placeholder the
        // transform substituted into the code is replaced with the emitted
        // file's public URL. Names derive from the key, so both sides agree by
        // construction. A worker whose own graph spawns workers nests one
        // level at a time; a runaway cycle is cut by the depth guard.
        let mut worker_urls: Vec<(String, String)> = Vec::new();
        {
            thread_local! {
                static WORKER_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
            }
            let mut worker_entries: Vec<(String, PathBuf)> = Vec::new();
            for &dense in &reachable_dense {
                if let Some(module) = self.graph.modules[dense].as_ref() {
                    for (key, entry) in &module.workers {
                        if !worker_entries.iter().any(|(existing, _)| existing == key) {
                            worker_entries.push((key.clone(), entry.clone()));
                        }
                    }
                }
            }
            if !worker_entries.is_empty() {
                let depth = WORKER_DEPTH.with(|cell| cell.get());
                if depth >= 4 {
                    return Err(format!(
                        "worker bundling nested {depth} levels deep — a worker graph that \
                         spawns workers recursively is not supported"
                    ));
                }
                let assets_dir = parent.join("assets");
                fs::create_dir_all(&assets_dir)
                    .map_err(|error| format!("cannot create {}: {error}", assets_dir.display()))?;
                let mut emitted_paths: Vec<(PathBuf, String)> = Vec::new();
                for (key, entry) in worker_entries {
                    let stem = entry
                        .file_stem()
                        .and_then(|value| value.to_str())
                        .unwrap_or("worker");
                    // One bundle per resolved ENTRY: several creation sites of
                    // the same worker share one emitted file, so the name
                    // derives from the entry path, not the per-site key.
                    let file_name = format!(
                        "{stem}-{:08x}.worker.js",
                        content_hash(entry.to_string_lossy().as_bytes()) as u32
                    );
                    if let Some((_, existing)) =
                        emitted_paths.iter().find(|(path, _)| *path == entry)
                    {
                        worker_urls.push((
                            format!("__diffpack_worker__{key}__"),
                            format!("{}assets/{existing}", self.resolution_cache.base),
                        ));
                        continue;
                    }
                    emitted_paths.push((entry.clone(), file_name.clone()));
                    WORKER_DEPTH.with(|cell| cell.set(depth + 1));
                    let result = (|| -> Result<(), String> {
                        let (worker_bundler, update) = Bundler::discover_with_driver_policies(
                            &entry,
                            &self.config,
                            self.resolution_cache.providers.clone(),
                            DriverPolicies {
                                compiler: Arc::clone(&self.compiler),
                                special_modules: Arc::clone(&self.special_modules),
                                runtime: Arc::clone(&self.runtime_policy),
                                output: Arc::clone(&self.output_policy),
                                source: Arc::clone(&self.resolution_cache.source_policy),
                            },
                        )?;
                        for warning in partition_diagnostics(
                            &update.diagnostics,
                            &format!("worker entry {}", entry.display()),
                        )? {
                            eprintln!("warning: {warning}");
                        }
                        let worker_reachable = worker_bundler.reachable_modules_direct();
                        let worker_stats = worker_bundler.emit_with_options(
                            &worker_reachable,
                            &assets_dir.join(&file_name),
                            EmitOptions {
                                format: ModuleFormat::BrowserEsm,
                                ..options
                            },
                        )?;
                        // The nested emit's own assets would land under
                        // `assets/assets/` while their minted URLs claim
                        // `{base}assets/` — broken references. Refuse until
                        // worker asset graphs are really supported.
                        let worker_assets_dir = assets_dir.join("assets");
                        if worker_stats
                            .written
                            .iter()
                            .any(|path| path.starts_with(&worker_assets_dir))
                        {
                            return Err(format!(
                                "worker entry {} references assets/CSS; asset graphs \
                                 inside module workers are not supported yet",
                                entry.display()
                            ));
                        }
                        stats.written.extend(worker_stats.written);
                        Ok(())
                    })();
                    WORKER_DEPTH.with(|cell| cell.set(depth));
                    result?;
                    worker_urls.push((
                        format!("__diffpack_worker__{key}__"),
                        format!("{}assets/{file_name}", self.resolution_cache.base),
                    ));
                }
            }
        }
        let substitute_workers = |mut bundle: RenderedBundle| -> RenderedBundle {
            for (placeholder, url) in &worker_urls {
                if bundle.code.contains(placeholder.as_str()) {
                    bundle.code = bundle.code.replace(placeholder.as_str(), url);
                }
            }
            bundle
        };
        // Top-level await and import.meta both lower through the registry in
        // CommonJS output. ESM output preserves their native syntax.
        // Which modules must render as `async` factories. Empty (and free) unless
        // some reachable module actually top-level-`await`s; when one does, the
        // property propagates up every static import edge, exactly as an ES
        // module's "async evaluation" does.
        let async_stage = diffpack_core::build_profile::stage("emit/async-closure");
        let async_modules = self.async_module_closure(&reachable_dense, &runtime_ids)?;
        drop(async_stage);
        // The stylesheet pipeline (for a Tailwind app: candidate scan, source
        // read, compile — the react-server graph's second-largest cost) and the
        // JS chunk renders touch disjoint outputs and only read `&self`, so they
        // run side by side. The css written-set merges into `stats` afterwards,
        // before the caller's stale-file prune ever looks at it.
        let (css_written, render_result) = rayon::join(
            || -> Result<BTreeSet<PathBuf>, String> {
                if cancel.cancelled() {
                    cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
                    return Ok(BTreeSet::new());
                }
                let css_stage = diffpack_core::build_profile::stage("emit/css");
                let mut written = BTreeSet::new();
                if self.emit_css(&allowed, output, &mut written, cancel)? {
                    cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                drop(css_stage);
                Ok(written)
            },
            || -> Result<(), String> {
                let split_stage = diffpack_core::build_profile::stage("emit/render-split-chunks");
                // Each split chunk's render, worker substitution, and file write
                // is independent work against `&self` (the render cache is a
                // Mutex and every chunk writes its own path), so the whole set
                // runs across the pool. The per-chunk accumulators (live key,
                // rendered flag, written paths) are merged serially afterwards,
                // keeping `stats` deterministic.
                let split_results = self.frontend_pool.install(|| {
                    plans
                        .par_iter()
                        .map(|plan| -> Result<(u64, bool, BTreeSet<PathBuf>), String> {
                            // Checked per chunk, not only inside a render: most of a
                            // compaction pass is cache hits and file writes over a
                            // thousand chunks, and a pass made entirely of those must
                            // still stop when the developer types.
                            if cancelled.load(std::sync::atomic::Ordering::Relaxed)
                                || cancel.cancelled()
                            {
                                cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
                                return Ok((0, false, BTreeSet::new()));
                            }
                            let chunk_path = parent.join(&plan.file_name);
                            let prerequisites = plan
                                .prerequisites
                                .iter()
                                .map(|&index| format!("./{}", plans[index].file_name))
                                .collect::<Vec<_>>();
                            let (rendered, key, fresh) = self.render_chunk_cached(
                                &plan.members,
                                &plan.roots,
                                &chunk_names,
                                &chunk_files,
                                &runtime_ids,
                                &global_demands,
                                &prerequisites,
                                false,
                                flat_allowed,
                                &async_modules,
                                options.format,
                                options.minify,
                                options.source_map,
                                options.hmr,
                                &plan.file_name,
                                cancel,
                            )?;
                            // Cancelled mid-render: this chunk is simply not written.
                            // What is already on disk stays valid, and the caller keeps
                            // the graph's debt so a later quiet moment finishes the job.
                            let Some(rendered) = rendered else {
                                cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
                                return Ok((key, false, BTreeSet::new()));
                            };
                            let rendered = substitute_workers(rendered);
                            let mut written = BTreeSet::new();
                            self.write_rendered(rendered, &chunk_path, options, &mut written)?;
                            Ok((key, fresh, written))
                        })
                        .collect::<Result<Vec<_>, String>>()
                })?;
                for (key, fresh, written) in split_results {
                    live_keys.insert(key);
                    if fresh {
                        stats.rendered_chunks += 1;
                    }
                    stats.written.extend(written);
                }
                drop(split_stage);
                if cancelled.load(std::sync::atomic::Ordering::Relaxed) || cancel.cancelled() {
                    cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
                    return Ok(());
                }
                let main_stage = diffpack_core::build_profile::stage("emit/render-main-chunk");
                let (rendered, main_key, main_fresh) = self.render_chunk_cached(
                    &main_modules,
                    &[self.graph.entry],
                    &chunk_names,
                    &chunk_files,
                    &runtime_ids,
                    &global_demands,
                    &[],
                    true,
                    flat_allowed,
                    &async_modules,
                    options.format,
                    options.minify,
                    options.source_map,
                    options.hmr,
                    entry_name,
                    cancel,
                )?;
                live_keys.insert(main_key);
                let Some(rendered) = rendered else {
                    cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
                    drop(main_stage);
                    return Ok(());
                };
                if main_fresh {
                    stats.rendered_chunks += 1;
                }
                let rendered = substitute_workers(rendered);
                self.write_rendered(rendered, output, options, &mut stats.written)?;
                drop(main_stage);
                Ok(())
            },
        );
        stats.written.extend(css_written?);
        render_result?;
        stats.cancelled = cancelled.load(std::sync::atomic::Ordering::Relaxed);
        // A cancelled emit rendered a SUBSET of the live chunks, so its `live_keys` is
        // not the live set — evicting against it would throw away cache entries the
        // next attempt needs and make that attempt slower, which is the opposite of
        // what abandoning this one was for.
        if !stats.cancelled {
            self.evict_render_cache(&live_keys);
        }
        Ok(stats)
    }

    /// Which reachable modules must render as `async` factories: those that
    /// top-level-`await`, plus — transitively — every module that statically
    /// imports one. This is exactly the ES module spec's "async evaluation"
    /// propagation: a module importing an async module cannot finish evaluating
    /// until that import has settled, so its own evaluation is asynchronous too.
    ///
    /// Only import sites that can actually carry an `await` propagate, and every
    /// other way of reaching an async module is a hard error naming both modules
    /// rather than a bundle that silently reads a half-initialised namespace:
    ///
    ///  * a top-level `import`/`export ... from` lowers to a marked statement
    ///    (`/*__diffpack_import:"spec"__*/ns=require.esm("spec");`) at the top
    ///    level of the factory body, where `await` is legal once the factory is
    ///    `async` — this is the propagating case;
    ///  * `export * from "spec"` lowers to a top-level `__reExport(...)` call,
    ///    likewise awaitable;
    ///  * `export * as ns from "spec"` lowers to a LAZY getter, which cannot
    ///    await — hard error;
    ///  * a CommonJS `require("spec")` is a synchronous call that may sit inside
    ///    any function body — hard error (Node itself throws
    ///    `ERR_REQUIRE_ASYNC_MODULE` here);
    ///  * a dynamic `import("spec")` already yields a promise and is handled by
    ///    the runtime's `require.dynamic`, so it does NOT make its importer async.
    fn async_module_closure(
        &self,
        reachable: &[DenseModuleId],
        runtime_ids: &[Option<usize>],
    ) -> Result<AsyncModules, String> {
        propagate_async_modules(
            &AsyncGraphView {
                modules: &self.graph.modules,
                ids: &self.graph.ids,
                runtime_ids: Some(runtime_ids),
            },
            reachable,
        )
    }

    /// DETECTION-ONLY variant of `Self::async_module_closure`: which reachable
    /// modules evaluate asynchronously (top-level `await`, plus everything that
    /// statically imports one, transitively). Unlike the emit-time closure this
    /// never errors — non-propagating edge kinds (a bare `require`, a lazy
    /// namespace re-export) are simply skipped, because the caller is asking the
    /// question BEFORE emit precisely to rewrite such edges into legal ones. The
    /// next adapter uses it to decide which client-island pins must stay eager
    /// static imports (an async island cannot be evaluated on demand by the RSC
    /// seam's synchronous require).
    pub fn async_tainted_modules(&self, reachable: &BTreeSet<ModuleId>) -> HashSet<ModuleId> {
        let reachable_dense: Vec<DenseModuleId> = reachable
            .iter()
            .filter_map(|id| self.graph.indices.get(id.as_str()).copied())
            .collect();
        let async_modules = detect_async_modules(
            &AsyncGraphView {
                modules: &self.graph.modules,
                ids: &self.graph.ids,
                runtime_ids: None,
            },
            &reachable_dense,
        );
        reachable_dense
            .into_iter()
            .filter(|dense| async_modules.is_async(*dense))
            .map(|dense| self.graph.ids[dense].to_string())
            .collect()
    }

    /// Renders one chunk, consulting the per-chunk render cache: on a hit the
    /// cached bytes are returned verbatim (byte-identical to a fresh render) and
    /// `render_best` is skipped; on a miss the chunk is rendered, cached, and
    /// the returned flag is true. The chunk's key is returned so the caller can
    /// record it live (surviving the post-emit eviction) whether or not it was
    /// re-rendered. `&self` throughout, so chunks render in parallel.
    #[allow(clippy::too_many_arguments)]
    fn render_chunk_cached(
        &self,
        modules: &[DenseModuleId],
        roots: &[DenseModuleId],
        chunk_names: &HashMap<DenseModuleId, String>,
        chunk_files: &[String],
        runtime_ids: &[Option<usize>],
        global_demands: &[ExportDemand],
        prerequisites: &[String],
        is_main: bool,
        flat_allowed: bool,
        async_modules: &AsyncModules,
        format: ModuleFormat,
        minify: bool,
        source_map: bool,
        hmr: bool,
        chunk_name: &str,
        cancel: EmitCancel<'_>,
    ) -> Result<(Option<RenderedBundle>, u64, bool), String> {
        let key = self.chunk_render_key(
            modules,
            roots,
            is_main,
            chunk_names,
            runtime_ids,
            global_demands,
            prerequisites,
            flat_allowed,
            async_modules,
            format,
            minify,
            source_map,
            hmr,
        );
        // A cache hit is free and always honoured: the bytes exist, so handing them
        // back cannot delay anyone. Only a real render is abandoned.
        if let Some(hit) = self.render_cache.lock().unwrap().entries.get(&key) {
            return Ok((Some(hit.clone()), key, false));
        }
        let best_stage = diffpack_core::build_profile::stage("emit/render-best");
        let rendered = self.render_best(
            modules,
            roots,
            chunk_names,
            chunk_files,
            runtime_ids,
            global_demands,
            prerequisites,
            is_main,
            flat_allowed,
            async_modules,
            format,
            hmr,
            cancel,
        )?;
        drop(best_stage);
        let Some(mut bundle) = rendered else {
            return Ok((None, key, false));
        };
        // The readable mappings are what BOTH output shapes' maps are built from
        // (a minified chunk's map is composed from them), so this is the one place
        // to prove they describe the bytes that were actually rendered.
        if source_map {
            let validate_stage = diffpack_core::build_profile::stage("emit/validate-mappings");
            validate_mappings(&bundle.code, &bundle.mappings, chunk_name, |module| {
                self.graph.ids[module].to_string()
            })?;
            drop(validate_stage);
        }
        // Whitespace/syntax minification of the FINISHED chunk. The chunk's `code`
        // is already clean, valid JS (markers were consumed during render; it
        // passes `node --check` and runs in-browser), so a final per-chunk Oxc
        // codegen-minify pass has a safe insertion point that never touches the
        // marker-based linker. Minified bytes are stored in the cache under a key
        // that folds in `minify` and `source_map`, so a leaf edit re-minifies (and
        // re-composes the map for) exactly this chunk and reuses the rest verbatim.
        if minify {
            if source_map {
                // Compose the two maps we can honestly obtain: the readable ->
                // original module mappings (already on `bundle`) THROUGH the
                // minified -> readable map Oxc codegen emits for the re-print. The
                // result resolves a minified position back to the correct ORIGINAL
                // source file+region. The readable `mappings` no longer describe
                // the emitted bytes, so they are cleared in favour of `map_json`.
                let minify_stage = diffpack_core::build_profile::stage("emit/minify");
                let (minified, minified_map) =
                    minify_chunk_code_with_map(&bundle.code, chunk_name)?;
                drop(minify_stage);
                let compose_stage = diffpack_core::build_profile::stage("emit/compose-map");
                let composed = self.compose_source_map(
                    &bundle.mappings,
                    &minified_map,
                    chunk_name,
                    chunk_name,
                )?;
                drop(compose_stage);
                bundle.code = minified;
                bundle.mappings = Vec::new();
                bundle.map_json = Some(composed);
            } else {
                bundle.code = minify_chunk_code(&bundle.code, chunk_name)?;
                // The readable-render mappings no longer describe the minified
                // bytes and no map was requested; clear them rather than ship a
                // map that lies about positions.
                bundle.mappings = Vec::new();
            }
        }
        self.render_cache
            .lock()
            .unwrap()
            .entries
            .insert(key, bundle.clone());
        Ok((Some(bundle), key, true))
    }

    /// Evicts every cached chunk render whose key was not used in the emit that
    /// just finished, bounding the cache to the currently-live chunk set.
    fn evict_render_cache(&self, live_keys: &HashSet<u64>) {
        self.render_cache
            .lock()
            .unwrap()
            .entries
            .retain(|key, _| live_keys.contains(key));
    }

    /// A stable, collision-resistant key for one chunk's rendered bytes.
    ///
    /// It folds in everything `render_best` reads to produce this chunk: the
    /// ordered dense-module ids, each member module's transformed-content hash and
    /// its dependency structure, the render params that shape the bytes
    /// (`format`, `is_main`, the chunk root), and — restricted to the chunk's own
    /// members and the targets they reference — the `runtime_ids`, `chunk_names`,
    /// and aggregated export demands. It deliberately does NOT fold in the whole
    /// (graph-wide) `runtime_ids`/`global_demands` vectors: a leaf edit shifts
    /// neither for any chunk that excludes the leaf, so those chunks keep their key
    /// and are reused, while the one chunk containing the leaf sees its member
    /// hash change and is re-rendered.
    #[allow(clippy::too_many_arguments)]
    fn chunk_render_key(
        &self,
        modules: &[DenseModuleId],
        roots: &[DenseModuleId],
        is_main: bool,
        chunk_names: &HashMap<DenseModuleId, String>,
        runtime_ids: &[Option<usize>],
        global_demands: &[ExportDemand],
        prerequisites: &[String],
        flat_allowed: bool,
        async_modules: &AsyncModules,
        format: ModuleFormat,
        minify: bool,
        source_map: bool,
        hmr: bool,
    ) -> u64 {
        render_key(
            &RenderKeyView {
                modules: &self.graph.modules,
                runtime_ids,
                demands: global_demands,
                async_modules,
            },
            modules,
            chunk_names,
            RenderKeyOptions {
                format: format as u8,
                any_async: async_modules.any,
                hmr,
                minify,
                source_map,
                is_main,
                roots,
                prerequisites,
                flat_allowed,
            },
        )
    }

    #[doc(hidden)]
    pub fn chunk_plan(
        &self,
        allowed: &HashSet<DenseModuleId>,
        entry_file: &str,
    ) -> Result<Vec<ChunkPlan>, String> {
        let mut static_edges = vec![Vec::new(); self.graph.ids.len()];
        let mut dynamic_edges = vec![Vec::new(); self.graph.ids.len()];
        for (source, module) in self.graph.modules.iter().enumerate() {
            let Some(module) = module else { continue };
            for (_, target, demand) in &module.dependencies {
                if demand.deferred() {
                    dynamic_edges[source].push(*target);
                } else {
                    static_edges[source].push(*target);
                }
            }
        }
        let private_chunk_names = self
            .config
            .private_chunk_names
            .iter()
            .filter_map(|(id, file_name)| {
                self.graph
                    .indices
                    .get(id.as_str())
                    .copied()
                    .filter(|module| allowed.contains(module))
                    .map(|module| (module, file_name.clone()))
            })
            .collect();
        ChunkGraph {
            entry: self.graph.entry,
            module_ids: &self.graph.ids,
            allowed,
            static_edges: &static_edges,
            dynamic_edges: &dynamic_edges,
            private_chunk_names: &private_chunk_names,
        }
        .plan(entry_file)
    }

    /// Derives the client build's route -> chunk mapping for the manifest.
    ///
    /// `entry_file` is the entry chunk name (`client.js`); `base` is the URL base
    /// the chunks are served from (`/`). Each dynamic-import chunk that is a
    /// route's `?tsr-split=*` split is attributed to that route's TanStack id (the
    /// `createFileRoute` argument), so a route with several split properties lists
    /// all of its chunks. `__root__` maps to the entry chunk, which statically
    /// bundles the root route and all shared code.
    ///
    /// Chunk names are computed with the identical ordering
    /// [`Self::emit_with_options`] uses, so the recorded URLs are exactly the files
    /// emitted to disk. A non-route dynamic chunk (not a `?tsr-split`) is emitted
    /// as a chunk but is not a route preload, so it is not attributed here.
    pub fn integration_manifest_graph(
        &self,
        reachable: &BTreeSet<ModuleId>,
        entry_file: &str,
    ) -> Result<IntegrationManifestGraph, String> {
        // The manifest must describe the SAME chunk set emit produces, so refine
        // the reachable set through the identical dead-module elimination pass and
        // then read the chunk assignment off the identical plan.
        let reachable = self.live_modules(reachable);
        let allowed = reachable
            .iter()
            .filter_map(|id| self.graph.indices.get(id.as_str()).copied())
            .collect::<HashSet<_>>();
        let plans = self.chunk_plan(&allowed, entry_file)?;
        let reachable_dense = reachable
            .iter()
            .filter_map(|id| self.graph.indices.get(id.as_str()).copied())
            .collect::<Vec<_>>();
        let mut runtime_ids = vec![None; self.graph.ids.len()];
        for (runtime_id, &dense) in reachable_dense.iter().enumerate() {
            runtime_ids[dense] = Some(runtime_id);
        }
        let mut chunk_of = HashMap::new();
        for plan in &plans {
            for &member in &plan.members {
                chunk_of.insert(member, plan.file_name.clone());
            }
        }
        let modules = reachable_dense
            .iter()
            .filter_map(|&dense| {
                let module = self.graph.modules[dense].as_ref()?;
                Some(IntegrationManifestModule {
                    id: self.graph.ids[dense].to_string(),
                    source: module.source.to_string(),
                    runtime_id: runtime_ids[dense].expect("reachable module has runtime id"),
                    chunk: chunk_of.get(&dense).cloned(),
                })
            })
            .collect();
        let chunks = plans
            .iter()
            .enumerate()
            .map(|(index, plan)| IntegrationManifestChunk {
                roots: plan
                    .roots
                    .iter()
                    .map(|&root| self.graph.ids[root].to_string())
                    .collect(),
                load_order: chunk_load_order(&plans, index),
            })
            .collect();
        Ok(IntegrationManifestGraph {
            entry_file: entry_file.to_string(),
            modules,
            chunks,
        })
    }

    /// The whole-graph derivation an emit performs before it renders any chunk,
    /// memoized per entry chunk. See [`EmitPlan`].
    ///
    /// The first call derives it; every later call for the same entry reuses it,
    /// including across incremental rebuilds. That reuse is not just an
    /// optimization, it is the correctness condition for a hot update: a
    /// micro-chunk has to bind against the runtime the browser and the dev server's
    /// Node processes have ALREADY LOADED, and that runtime came from the last full
    /// emit. Re-deriving reads the CURRENT graph, and an edit that changes which
    /// exports are live changes the surviving module set — which renumbers every
    /// runtime id after it, so the micro-chunk would register its factories under
    /// ids the loaded runtime does not know.
    ///
    /// A change that really does re-partition the graph (a module added or removed
    /// from the reachable set) cannot be hot-patched at all: the dev server detects
    /// it as `graph_changed`, does a full rebuild, and re-emits — and that emit
    /// replaces this plan through [`Self::record_emit_plan`].
    fn emit_plan(
        &self,
        reachable: &BTreeSet<ModuleId>,
        entry_file: &str,
    ) -> Result<Arc<EmitPlan>, String> {
        if let Some(plan) = self
            .emit_plans
            .lock()
            .map_err(|_| "emit plan cache poisoned".to_string())?
            .get(entry_file)
        {
            return Ok(Arc::clone(plan));
        }
        let live = self.live_modules(reachable);
        let reachable_dense = live
            .iter()
            .filter_map(|id| self.graph.indices.get(id.as_str()).copied())
            .collect::<Vec<_>>();
        let allowed = reachable_dense.iter().copied().collect::<HashSet<_>>();
        let plans = self.chunk_plan(&allowed, entry_file)?;
        let plan = Arc::new(self.build_emit_plan(reachable_dense, allowed, &plans));
        self.record_emit_plan(entry_file, Arc::clone(&plan))?;
        Ok(plan)
    }

    /// Assemble an [`EmitPlan`] from the pieces an emit already computed, so the
    /// stored plan is literally the one that rendered the bundle now on disk.
    #[doc(hidden)]
    pub fn build_emit_plan(
        &self,
        reachable_dense: Vec<DenseModuleId>,
        allowed: HashSet<DenseModuleId>,
        plans: &[ChunkPlan],
    ) -> EmitPlan {
        let mut runtime_ids = vec![None; self.graph.ids.len()];
        for (runtime_id, &dense) in reachable_dense.iter().enumerate() {
            runtime_ids[dense] = Some(runtime_id);
        }
        let mut chunk_of: HashMap<DenseModuleId, String> = HashMap::new();
        for plan in plans {
            for &member in &plan.members {
                chunk_of.insert(member, plan.file_name.clone());
            }
        }
        EmitPlan {
            chunk_names: chunk_names(plans),
            chunk_files: plans.iter().map(|plan| plan.file_name.clone()).collect(),
            reachable_dense,
            allowed,
            runtime_ids,
            chunk_of,
        }
    }

    /// Publish the plan an emit just used as the one every later HMR micro-chunk
    /// for that entry must agree with.
    fn record_emit_plan(&self, entry_file: &str, plan: Arc<EmitPlan>) -> Result<(), String> {
        self.emit_plans
            .lock()
            .map_err(|_| "emit plan cache poisoned".to_string())?
            .insert(entry_file.to_string(), plan);
        Ok(())
    }

    /// DEV-ONLY: for each changed module id, its stable runtime id and the chunk
    /// file that (re-)registers its factory, so the dev server can push a targeted
    /// HMR update. Reads [`Self::emit_with_options`]'s own runtime-id and chunk
    /// assignment, so the ids/chunks match the bytes actually emitted. A module
    /// not in the live set (e.g. tree-shaken away) is skipped.
    pub fn hmr_locate(
        &self,
        reachable: &BTreeSet<ModuleId>,
        changed: &BTreeSet<ModuleId>,
        entry_file: &str,
    ) -> Result<Vec<HmrLocation>, String> {
        // The same partition emit used, so the chunk a module is reported in is
        // the chunk whose bytes actually carry its factory. Membership is
        // disjoint, so this is a plain lookup rather than a preference order: a
        // module is in exactly one plan chunk, or in the entry chunk.
        let plan = self.emit_plan(reachable, entry_file)?;
        let mut located = Vec::new();
        for module_id in changed {
            let Some(&dense) = self.graph.indices.get(module_id.as_str()) else {
                continue;
            };
            let Some(runtime_id) = plan.runtime_ids[dense] else {
                continue;
            };
            let chunk_file = plan
                .chunk_of
                .get(&dense)
                .map_or_else(|| entry_file.to_string(), Clone::clone);
            located.push(HmrLocation {
                module_id: module_id.to_string(),
                runtime_id,
                chunk_file,
            });
        }
        Ok(located)
    }

    /// Render a TINY standalone HMR chunk carrying ONLY the `changed` modules, in the
    /// split-chunk (register-only) format. A browser Fast Refresh imports this (~1 KB)
    /// instead of re-importing and re-parsing the whole entry chunk (which bundles the
    /// entire app + React — ~1 MB — so re-parsing it dominates the browser-side hot
    /// update). Runtime ids, export demands and chunk names are reconstructed exactly
    /// as [`Self::emit_with_options`] computed them for the live emit, so the freshly
    /// registered factory binds against the same global runtime and the same target
    /// ids already loaded from the entry chunk. `roots` is empty, so the chunk ends in
    /// `return __runtime;` and only REGISTERS (never re-evaluates) — the browser then
    /// drives the swap through `hmrApply`. Returns `None` if no changed module is live.
    ///
    /// The chunk is WRITTEN here rather than returned as text, because it needs the
    /// same source-map sidecar every other emitted chunk gets: this is the code the
    /// developer is editing right now, so a hot update that lands in the browser
    /// with no map — which is what happened before — is the most user-visible way
    /// for a stack trace to become unreadable. `path` names the chunk file; the map
    /// goes beside it under the shared `Self::source_map_sidecar` naming.
    ///
    /// `format` MUST be the format the graph's own emit used
    /// ([`ModuleFormat::BrowserEsm`] for the client, [`ModuleFormat::Esm`] for a Node
    /// server graph). It is not cosmetic: a module that references `__dirname` /
    /// `__filename` renders with BROWSER stubs (`"/index.js"`, `"/"`) under
    /// `BrowserEsm` and binds the Node ESM entry's real values under `Esm`, so
    /// rendering a server micro-chunk as browser output would silently swap a server
    /// module's file paths for stubs the moment it was hot-updated.
    pub fn write_hmr_chunk(
        &self,
        reachable: &BTreeSet<ModuleId>,
        changed: &BTreeSet<ModuleId>,
        entry_file: &str,
        options: EmitOptions,
        format: ModuleFormat,
        path: &Path,
    ) -> Result<bool, String> {
        let Some(rendered) =
            self.render_hmr_chunk(reachable, changed, entry_file, options, format)?
        else {
            return Ok(false);
        };
        let sidecar = if options.source_map {
            Some(self.source_map_sidecar(&rendered, path)?)
        } else {
            None
        };
        let mut code = rendered.code;
        if let Some((map_path, contents, comment)) = sidecar {
            // Written BEFORE the chunk: the browser fetches the map only after it has
            // the chunk, so this order can never expose a `sourceMappingURL` whose
            // target is not on disk yet.
            std::fs::write(&map_path, contents.as_bytes()).map_err(|error| {
                format!(
                    "cannot write hmr source map {}: {error}",
                    map_path.display()
                )
            })?;
            code.push_str(&comment);
        }
        std::fs::write(path, code.as_bytes())
            .map_err(|error| format!("cannot write hmr chunk {}: {error}", path.display()))?;
        Ok(true)
    }

    /// Render the HMR micro-chunk. See [`Self::write_hmr_chunk`], which is what the
    /// dev server calls — this returns the un-written bundle (code + its real
    /// per-module mappings) so the caller can attach the map sidecar.
    fn render_hmr_chunk(
        &self,
        reachable: &BTreeSet<ModuleId>,
        changed: &BTreeSet<ModuleId>,
        entry_file: &str,
        options: EmitOptions,
        format: ModuleFormat,
    ) -> Result<Option<RenderedBundle>, String> {
        let plan = self.emit_plan(reachable, entry_file)?;
        let reachable_dense = &plan.reachable_dense;
        let runtime_ids = &plan.runtime_ids;
        // Only live modules can be rendered; a changed-but-tree-shaken module is
        // dropped (the caller's `hmr_locate` likewise skips it).
        let changed_dense = changed
            .iter()
            .filter_map(|id| self.graph.indices.get(id.as_str()).copied())
            .filter(|dense| plan.allowed.contains(dense))
            .collect::<Vec<_>>();
        if changed_dense.is_empty() {
            return Ok(None);
        }
        // Export demand and the async-factory closure are re-derived from the CURRENT
        // graph rather than taken from the plan: both are content-derived, and the
        // module being hot-updated is exactly the module whose content just changed.
        // They are also cheap (~2 ms each on cal.com) next to the plan's ~37 ms.
        let global_demands = self.export_demands(reachable_dense);
        let async_modules = self.async_module_closure(reachable_dense, runtime_ids)?;
        let bundle = self.render_best(
            &changed_dense,
            &[], // no roots -> the tail is `return __runtime;` (register-only)
            &plan.chunk_names,
            &plan.chunk_files,
            runtime_ids,
            &global_demands,
            &[],   // no prerequisite chunk loads
            false, // not the main chunk (no runtime bootstrap)
            false, // registry format, never scope-hoisted flat
            &async_modules,
            format,
            options.hmr,
            // A hot micro-chunk IS the edit's payload, never deferred housekeeping:
            // there is nothing it could usefully be abandoned for.
            EmitCancel::never(),
        )?;
        Ok(bundle)
    }

    /// Copies every content-hashed asset referenced by a reachable module into
    /// `<output_dir>/assets/`. Deduplicated by public name, so an asset imported
    /// from several modules is written once.
    fn emit_assets(
        &self,
        allowed: &HashSet<DenseModuleId>,
        parent: &Path,
        written: &mut BTreeSet<PathBuf>,
        cancel: EmitCancel<'_>,
    ) -> Result<(), String> {
        let mut seen = HashSet::new();
        let mut assets_dir_ready = false;
        for &dense in allowed {
            let Some(module) = self.graph.modules[dense].as_ref() else {
                continue;
            };
            for asset in &module.assets {
                if !seen.insert(asset.public_name.clone()) {
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
                if let Some(source_css) = &asset.tailwind_source {
                    // Compile the Tailwind v4 entry against the class candidates
                    // scanned from the app's source tree (the scan root the entry
                    // declares via `source(...)`). This is a build-emit step, off
                    // the incremental transform hot path.
                    let compiled =
                        self.compile_tailwind_entry(&asset.source, source_css, parent, cancel)?;
                    // A `?url` sheet is part of the emit's asset pass, which the caller
                    // has already gated on the same signal; an abandoned compile here
                    // simply leaves the previous file in place.
                    let Some(compiled) = compiled else {
                        return Ok(());
                    };
                    write_if_changed(&destination, compiled.as_bytes())?;
                } else if !destination.exists() {
                    // The public name is content-hashed, so a destination that
                    // already exists holds identical bytes and needs no recopy.
                    fs::copy(&asset.source, &destination).map_err(|error| {
                        format!(
                            "cannot copy asset {} to {}: {error}",
                            asset.source.display(),
                            destination.display()
                        )
                    })?;
                }
                written.insert(destination);
                // Next static-image import: additionally emit the responsive
                // downscale variants next to the copied original (the SAME native
                // resize the public-image path uses, done once at emit time — off
                // the transform hot path, mirroring the `tailwind_source` case).
                // Variant names are content-hashed via `public_name`, so an
                // already-written variant needs no re-encode.
                if let Some(widths) = &asset.image_variants {
                    let _stage = diffpack_core::build_profile::stage("emit/assets-image-variants");
                    let decoded = image::open(&asset.source).map_err(|error| {
                        format!("cannot decode image {}: {error}", asset.source.display())
                    })?;
                    let (intrinsic_w, intrinsic_h) = (decoded.width().max(1), decoded.height());
                    for &width in widths {
                        let variant_name = asset_variant_public_name(&asset.public_name, width);
                        let variant_dest = assets_dir.join(&variant_name);
                        if variant_dest.exists() {
                            written.insert(variant_dest);
                            continue;
                        }
                        let target_h = ((intrinsic_h as u64 * width as u64) / intrinsic_w as u64)
                            .max(1) as u32;
                        let variant =
                            decoded.resize(width, target_h, image::imageops::FilterType::Triangle);
                        variant.save(&variant_dest).map_err(|error| {
                            format!("cannot write {}: {error}", variant_dest.display())
                        })?;
                        written.insert(variant_dest);
                    }
                }
            }
            for asset in &module.provider_assets {
                let public_name = asset
                    .name
                    .as_deref()
                    .and_then(|name| Path::new(name).file_name())
                    .and_then(|name| name.to_str())
                    .filter(|name| !name.is_empty())
                    .map(str::to_owned)
                    .unwrap_or_else(|| {
                        format!("provider-{:016x}.bin", content_hash(&asset.source))
                    });
                if !seen.insert(public_name.clone()) {
                    continue;
                }
                let assets_dir = parent.join("assets");
                if !assets_dir_ready {
                    fs::create_dir_all(&assets_dir).map_err(|error| {
                        format!("cannot create {}: {error}", assets_dir.display())
                    })?;
                    assets_dir_ready = true;
                }
                let destination = assets_dir.join(public_name);
                write_if_changed(&destination, &asset.source)?;
                written.insert(destination);
            }
        }
        Ok(())
    }

    /// Compiles one Tailwind entry — the single place both emit paths (a global
    /// `import "./app.css"` and a `?url` asset) go through, so they can never pick
    /// different engines for the same sheet.
    ///
    /// The native engine is the default and stays the whole performance story: an
    /// app using only what diffpack implements compiles in-process, with no `node`
    /// spawn and no `node_modules` read. A sheet that needs something the native
    /// engine does not implement — a JavaScript `@plugin`, an unknown at-rule, an
    /// `@apply` of a plugin-registered utility — is handed WHOLE to the app's own
    /// `tailwindcss` (see [`crate::tailwind_delegate`]). Which of the two ran is
    /// always reported.
    ///
    /// `css` is the entry's spliced text: its `@import`s are already inlined, so an
    /// `@apply`/`@utility`/`@plugin` written in an imported file is part of this
    /// compile for both engines.
    fn compile_tailwind_entry(
        &self,
        css_path: &Path,
        css: &str,
        out_root: &Path,
        cancel: EmitCancel<'_>,
    ) -> Result<Option<String>, String> {
        let scan_root = tailwind_scan_root(css_path, css);
        let candidates_stage = diffpack_core::build_profile::stage("css/tailwind-candidate-scan");
        // What the scan actually depends on: where it walks and what it skips. The
        // entry's remaining text (rules, theme, `@plugin`) does not enter it, so an
        // edit to the sheet keeps the scan — which is the whole point, since that walk
        // is the expensive half of compiling a monorepo's stylesheet.
        let globs = tailwind_source_globs(css)?;
        let key = (
            scan_root.clone(),
            out_root.to_path_buf(),
            format!("{globs:?}"),
        );
        let cached = self
            .tailwind_scan_cache
            .lock()
            .unwrap()
            .get(&key)
            .map(TailwindScan::candidates);
        let candidates = match cached {
            Some(hit) => hit,
            None => {
                let Some(per_file) =
                    self.tailwind_scan_files(&scan_root, out_root, &globs, cancel)?
                else {
                    return Ok(None);
                };
                let scan = TailwindScan { per_file };
                let candidates = scan.candidates();
                self.tailwind_scan_cache.lock().unwrap().insert(key, scan);
                candidates
            }
        };
        drop(candidates_stage);
        let theme_stage = diffpack_core::build_profile::stage("css/tailwind-app-theme");
        let app_theme = app_tailwind_theme_full(&scan_root, css, css_path);
        drop(theme_stage);
        // The compile is a pure function of (entry text, candidates, theme), so an
        // identical request is answered from the last few results instead of spawning
        // the app's Tailwind again. This is what keeps the dev loop's deferred chunk
        // compaction from recompiling the sheet the edit already compiled.
        let sheet_key = TailwindSheetKey {
            css: css.to_string(),
            candidates: hash_of(&candidates),
            theme: hash_of(&app_theme),
        };
        if let Some((_, hit)) = self
            .tailwind_sheet_cache
            .lock()
            .unwrap()
            .iter()
            .find(|(key, _)| *key == sheet_key)
        {
            return Ok(Some(hit.to_string()));
        }
        let Some(compiled) =
            self.compile_tailwind_uncached(css_path, css, &candidates, app_theme, cancel)?
        else {
            return Ok(None);
        };
        {
            /// How many compiled sheets to keep. A dev session recompiles ONE entry as
            /// it is edited, so the useful window is the newest few; each sheet is
            /// hundreds of KB, so the window is small on purpose.
            const KEEP: usize = 4;
            let mut cache = self.tailwind_sheet_cache.lock().unwrap();
            cache.push((sheet_key, Arc::new(compiled.clone())));
            let excess = cache.len().saturating_sub(KEEP);
            cache.drain(..excess);
        }
        Ok(Some(compiled))
    }

    /// The Tailwind compile itself: pick the engine and run it. Separated from
    /// [`Self::compile_tailwind_entry`] only so the cache in that function wraps one
    /// call rather than two engine branches.
    fn compile_tailwind_uncached(
        &self,
        css_path: &Path,
        css: &str,
        candidates: &BTreeSet<String>,
        app_theme: Option<String>,
        cancel: EmitCancel<'_>,
    ) -> Result<Option<String>, String> {
        let _compile_stage = diffpack_core::build_profile::stage("css/tailwind-compile");
        match diffpack_default_loader::tailwind::native_gap(css, app_theme.as_deref()) {
            Some(gap) => {
                let Some(sheet) = diffpack_default_loader::tailwind_delegate::compile(
                    css_path, css, candidates, &gap, cancel,
                )?
                else {
                    return Ok(None);
                };
                report_tailwind_engine(
                    css_path,
                    &format!(
                        "delegated to the app's own tailwindcss v{} (via {}) because {gap}. \
                         The app's engine is authoritative for this sheet, so diffpack's \
                         vendored v{} Tailwind data does not apply to it.",
                        sheet.version,
                        sheet.engine,
                        diffpack_default_loader::tailwind::VERSION,
                    ),
                );
                Ok(Some(sheet.css))
            }
            None => {
                // Only a NATIVE compile mixes the app's installed theme with
                // diffpack's vendored preflight, so only it can drift.
                if let Some(package) = installed_tailwind_dir(css_path) {
                    warn_on_tailwind_version_drift(&package);
                }
                report_tailwind_engine(
                    css_path,
                    &format!(
                        "compiled natively by diffpack's Tailwind engine (v{} data, no node)",
                        diffpack_default_loader::tailwind::VERSION
                    ),
                );
                diffpack_default_loader::tailwind::compile_with_theme_lenient(
                    css,
                    candidates,
                    app_theme.as_deref(),
                )
                .map(Some)
            }
        }
    }

    /// Scans the class candidates a Tailwind entry compiles against. Tailwind v4
    /// scans a source root (declared via `@import 'tailwindcss' source('..')`,
    /// resolved relative to the entry file); every JS/TS/JSX file under it
    /// contributes its `className`/`class` tokens. Falls back to the entry's own
    /// directory when no `source(...)` is given.
    fn tailwind_scan_files(
        &self,
        scan_root: &Path,
        out_root: &Path,
        // `@source` widens the scan beyond the project root — the way a monorepo
        // app declares that its classes also live in sibling workspace packages —
        // and `@source not` narrows it. Both are inputs to THIS scan, read off the
        // (already import-spliced, path-absolutized) entry text by the caller, which
        // also keys its scan cache on them.
        globs: &(Vec<String>, Vec<String>),
        cancel: EmitCancel<'_>,
    ) -> Result<Option<HashMap<PathBuf, diffpack_default_loader::tailwind::SourceScan>>, String>
    {
        let mut skip = ScanSkip::for_root(scan_root, out_root);
        let (included, excluded) = globs;
        skip.set_excluded(excluded);
        let walk_stage = diffpack_core::build_profile::stage("css/tailwind-walk-sources");
        let mut paths = Vec::new();
        // The walk enumerates every source file under the root — a monorepo has thousands —
        // so it asks between directories whether it is still wanted.
        if !collect_scan_sources(scan_root, &mut paths, &skip, &cancel) {
            return Ok(None);
        }
        for pattern in included {
            collect_glob_sources(pattern, &mut paths, &skip);
            if cancel.cancelled() {
                return Ok(None);
            }
        }
        drop(walk_stage);
        // READ + TOKENIZE IN PARALLEL. Both are per-file independent work over thousands of
        // files, and serially they were the single largest fixed cost of a dev cold start on
        // cal.com (611 ms reading, 291 ms tokenizing, on a 12-core machine). The result is a
        // path-keyed map, so the order files are visited in cannot change it. A file that
        // cannot be read is skipped exactly as before.
        let read_stage = diffpack_core::build_profile::stage("css/tailwind-read-and-scan-sources");
        let per_file: HashMap<PathBuf, diffpack_default_loader::tailwind::SourceScan> = paths
            .into_par_iter()
            .filter(|_| !cancel.cancelled())
            .filter_map(|path| {
                let source = fs::read_to_string(&path).ok()?;
                Some((
                    path,
                    diffpack_default_loader::tailwind::scan_source_parts(&source),
                ))
            })
            .collect();
        drop(read_stage);
        // Cancelled part-way through: the map is missing whatever the remaining files would
        // have contributed, and a partial candidate set compiles a stylesheet that silently
        // lacks utilities. Report "no scan" instead, exactly as the walk does.
        if cancel.cancelled() {
            return Ok(None);
        }
        Ok(Some(per_file))
    }

    /// Re-tokenize ONE file in every cached scan that covers it, and update the union.
    /// Called by the dev loop for each changed source file, in place of dropping the
    /// whole scan.
    pub fn refresh_tailwind_scan_path(&self, path: &Path) {
        let mut cache = self.tailwind_scan_cache.lock().unwrap();
        for scan in cache.values_mut() {
            if !scan.per_file.contains_key(path) {
                continue;
            }
            match fs::read_to_string(path) {
                Ok(source) => {
                    scan.per_file.insert(
                        path.to_path_buf(),
                        diffpack_default_loader::tailwind::scan_source_parts(&source),
                    );
                }
                // Gone: it contributes nothing now.
                Err(_) => {
                    scan.per_file.remove(path);
                }
            }
        }
    }

    /// Extracts the stylesheet: concatenates every reachable global CSS module's
    /// text in module execution order and writes it beside the bundle as
    /// `<output_stem>.css`. Nothing is written when no CSS is imported.
    ///
    /// NO source map is emitted for the stylesheet, and this is deliberate rather
    /// than an oversight. `module.css` is a bare `String` by the time it reaches
    /// here: every stage that produced it — the Sass/Less/Stylus compile, the
    /// PostCSS pass (run with `map: false`), the CSS-modules class rename, the
    /// nested-selector flattening, `@import` inlining, `url()` rewriting to hashed
    /// asset names, and the native Tailwind compile (whose utilities are
    /// SYNTHESIZED from scanned class candidates and have no author position at
    /// all) — returns text and nothing else. Writing a map here would mean
    /// inventing positions for text whose provenance this bundler does not have,
    /// which is precisely what [`crate::source_map`] exists to prevent. Emitting a
    /// real one means consuming each delegated tool's own map and composing the
    /// chain, the CSS-side equivalent of the JS work in `crate::source_map`.
    fn emit_css(
        &self,
        allowed: &HashSet<DenseModuleId>,
        output: &Path,
        written: &mut BTreeSet<PathBuf>,
        cancel: EmitCancel<'_>,
    ) -> Result<bool, String> {
        let order = self
            .static_execution_order(self.graph.entry, allowed)
            .unwrap_or_else(|| {
                let mut ids = allowed.iter().copied().collect::<Vec<_>>();
                ids.sort_by(|left, right| self.graph.ids[*left].cmp(&self.graph.ids[*right]));
                ids
            });
        // Remote `@import`s (scheme URLs that cannot be inlined) are hoisted,
        // deduped, to the very top: an @import is only valid before all rules,
        // so leaving one at its source position in the concatenation would be
        // silently ignored by the browser.
        let mut stylesheet = String::new();
        let mut hoisted = BTreeSet::new();
        for dense in &order {
            if let Some(module) = self.graph.modules[*dense].as_ref() {
                for external in &module.css_external_imports {
                    if hoisted.insert(external.clone()) {
                        stylesheet.push_str(external);
                        stylesheet.push('\n');
                    }
                }
            }
        }
        for dense in order {
            if let Some(module) = self.graph.modules[dense].as_ref()
                && let Some(css) = &module.css
            {
                if !stylesheet.is_empty() && !stylesheet.ends_with('\n') {
                    stylesheet.push('\n');
                }
                // A globally-imported Tailwind v4 entry carries its RAW source;
                // compile it here against freshly-scanned class candidates,
                // exactly as the `?url` asset path does in `emit_assets`.
                if diffpack_default_loader::tailwind::needs_native_tailwind_compile(css) {
                    let css_path = Path::new(self.graph.ids[dense].as_ref());
                    let out_root = output.parent().unwrap_or_else(|| Path::new("."));
                    let Some(compiled) =
                        self.compile_tailwind_entry(css_path, css, out_root, cancel)?
                    else {
                        // Abandoned: a partial sheet must never be written — it would
                        // serve a page with most of its utilities missing.
                        return Ok(true);
                    };
                    stylesheet.push_str(&compiled);
                } else {
                    stylesheet.push_str(css);
                }
            }
        }
        if stylesheet.is_empty() {
            // No stylesheet this emit; leaving it out of `written` lets the caller
            // prune a stale `.css` left by a previous build.
            return Ok(false);
        }
        let css_path = output.with_extension("css");
        write_if_changed(&css_path, stylesheet.as_bytes())?;
        written.insert(css_path);
        Ok(false)
    }

    /// The map file a chunk's map goes in, its contents, and the trailing comment
    /// that points the debugger at it.
    ///
    /// Every writer of an emitted chunk goes through here — the production emit
    /// and the dev HMR micro-chunk alike — so the sidecar's NAME and the URL in
    /// the chunk are derived from one place and cannot drift into a
    /// `sourceMappingURL` with no file behind it (which costs the browser a failed
    /// fetch on every load and tells the developer nothing).
    fn source_map_sidecar(
        &self,
        rendered: &RenderedBundle,
        output: &Path,
    ) -> Result<(PathBuf, String, String), String> {
        let map_path = path_with_suffix(output, ".map");
        let map_name = map_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| format!("source-map path is not UTF-8: {}", map_path.display()))?
            .to_owned();
        let output_name = output
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| format!("output path is not UTF-8: {}", output.display()))?;
        // A minified chunk carries a pre-composed map (`map_json`, minified ->
        // original); a readable chunk builds its map from the `ModuleMapping`
        // list (readable-generated -> original) at write time. Both resolve a
        // position in the emitted bytes to the correct original source.
        // The `sourceMappingURL` comment is part of the emitted file, so the map
        // has to cover the lines it adds too — otherwise those last lines carry no
        // segment and a consumer resolves them back into the final module's code.
        let comment = format!("\n//# sourceMappingURL={map_name}\n");
        let contents = match &rendered.map_json {
            Some(json) => json.clone(),
            None => {
                let mut emitted = rendered.code.clone();
                emitted.push_str(&comment);
                self.source_map(&rendered.mappings, output_name, &emitted)
            }
        };
        Ok((map_path, contents, comment))
    }

    fn write_rendered(
        &self,
        rendered: RenderedBundle,
        output: &Path,
        options: EmitOptions,
        written: &mut BTreeSet<PathBuf>,
    ) -> Result<(), String> {
        let sidecar = if options.source_map {
            Some(self.source_map_sidecar(&rendered, output)?)
        } else {
            None
        };
        let mut code = rendered.code;
        if let Some((map_path, contents, comment)) = sidecar {
            write_if_changed(&map_path, contents.as_bytes())?;
            written.insert(map_path);
            code.push_str(&comment);
        }
        // Skip the write when the on-disk bytes already match, so an unchanged,
        // cache-reused chunk is not needlessly rewritten (atomic per-file, only
        // the changed chunk touches disk).
        write_if_changed(output, code.as_bytes())?;
        written.insert(output.to_path_buf());
        Ok(())
    }

    fn static_closure(
        &self,
        root: DenseModuleId,
        allowed: &HashSet<DenseModuleId>,
    ) -> Vec<DenseModuleId> {
        static_closure(&StaticModuleGraphView { graph: &self.graph }, root, allowed)
    }

    pub fn all_modules(&self) -> BTreeSet<ModuleId> {
        self.graph
            .modules
            .iter()
            .enumerate()
            .filter(|(_, module)| module.is_some())
            .map(|(index, _)| self.graph.ids[index].to_string())
            .collect()
    }

    /// Builds a persistent dense reachability index for incremental edits.
    pub fn direct_reachability(&self) -> DirectReachability {
        let modules = self
            .graph
            .modules
            .iter()
            .enumerate()
            .filter(|(_, module)| module.is_some())
            .map(|(index, _)| self.graph.ids[index].to_string());
        let edges = self
            .graph
            .modules
            .iter()
            .enumerate()
            .flat_map(|(source, module)| {
                module.into_iter().flat_map(move |module| {
                    module.dependencies.iter().map(move |(_, target, _)| {
                        (
                            self.graph.ids[source].to_string(),
                            self.graph.ids[*target].to_string(),
                        )
                    })
                })
            });
        DirectReachability::new(self.graph.ids[self.graph.entry].to_string(), modules, edges)
    }

    /// Recomputes entry reachability from scratch using dense integer IDs.
    pub fn reachable_modules_direct(&self) -> BTreeSet<ModuleId> {
        self.direct_reachability().reachable_modules()
    }

    pub fn visualization_graph(&self, reachable: &BTreeSet<ModuleId>) -> VisualizationGraph {
        let nodes = self
            .graph
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
                    id: self.graph.ids[dense_id].to_string(),
                    dense_id,
                    reachable: reachable.contains(self.graph.ids[dense_id].as_ref()),
                    is_entry: dense_id == self.graph.entry,
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
                                format!("{name} = {}", expression.to_javascript())
                            })
                            .collect()
                    }),
                    foldable_effects: foldable.map_or_else(Vec::new, |foldable| {
                        foldable
                            .console_logs
                            .iter()
                            .map(|expression| {
                                format!("console.log({})", expression.to_javascript())
                            })
                            .collect()
                    }),
                    pruned_imports,
                })
            })
            .collect::<Vec<_>>();
        let edges = self
            .graph
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
            entry: self.graph.ids[self.graph.entry].to_string(),
            nodes,
            edges,
        }
    }

    pub fn watch_root(&self) -> PathBuf {
        PathBuf::from(self.graph.ids[self.graph.entry].as_ref())
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
        diagnostics: &mut Vec<Diagnostic>,
        record_delta: bool,
    ) -> Result<usize, String> {
        // Pipelined discovery: a module's dependencies are spawned the moment
        // its own load finishes — no wave barrier, so workers never idle while
        // a breadth-first frontier drains (the previous fork-join-per-wave
        // structure capped a 32-core machine at ~3x parallelism on a fan-out
        // graph, because early waves have 1-4 modules and every wave ends in a
        // full join). A shared seen-set dedups spawns; results are interned
        // afterwards in sorted-path order, so dense ids — and therefore every
        // downstream byte — stay deterministic run to run.
        type LoadResults = Vec<(SharedModuleId, Result<LoadedModule, String>)>;
        struct DiscoverShared<'bundler> {
            resolver: &'bundler Resolvers,
            resolution_cache: &'bundler ResolutionCache,
            compiler: &'bundler dyn ModuleCompiler,
            special_modules: &'bundler dyn SpecialModulePolicy,
            target: Target,
            hmr: bool,
            source_maps: bool,
            indices: &'bundler HashMap<SharedModuleId, DenseModuleId>,
            modules: &'bundler Vec<Option<ModuleState>>,
            state: std::sync::Mutex<(HashSet<SharedModuleId>, LoadResults)>,
        }
        impl DiscoverShared<'_> {
            fn already_loaded(&self, path: &str) -> bool {
                self.indices
                    .get(path)
                    .is_some_and(|index| self.modules[*index].is_some())
            }
        }
        fn spawn_load<'scope>(
            scope: &rayon::Scope<'scope>,
            shared: &'scope DiscoverShared<'scope>,
            path: SharedModuleId,
        ) {
            scope.spawn(move |scope| {
                let result = load_uncached(
                    shared.resolver,
                    shared.resolution_cache,
                    shared.compiler,
                    shared.special_modules,
                    Path::new(path.as_ref()),
                    shared.target,
                    shared.hmr,
                    shared.source_maps,
                );
                let mut next = Vec::new();
                {
                    let mut guard = shared.state.lock().unwrap();
                    if let Ok(loaded) = &result {
                        for (_, target, _) in &loaded.dependencies {
                            if !shared.already_loaded(target) && guard.0.insert(target.clone()) {
                                next.push(target.clone());
                            }
                        }
                    }
                    guard.1.push((path, result));
                }
                for target in next {
                    spawn_load(scope, shared, target);
                }
            });
        }

        let initial = paths
            .into_iter()
            .filter(|path| {
                self.graph
                    .indices
                    .get(path.as_ref())
                    .is_none_or(|index| self.graph.modules[*index].is_none())
            })
            .collect::<BTreeSet<_>>();
        let shared = DiscoverShared {
            resolver: &self.resolver,
            resolution_cache: &self.resolution_cache,
            compiler: self.compiler.as_ref(),
            special_modules: self.special_modules.as_ref(),
            target: self.target,
            hmr: self.hmr,
            source_maps: self.config.source_maps,
            indices: &self.graph.indices,
            modules: &self.graph.modules,
            state: std::sync::Mutex::new((initial.iter().cloned().collect(), Vec::new())),
        };
        self.frontend_pool.install(|| {
            rayon::scope(|scope| {
                for path in initial {
                    spawn_load(scope, &shared, path);
                }
            });
        });
        let mut loaded = shared.state.into_inner().unwrap().1;
        // Sorted-path interning keeps dense-id assignment independent of
        // completion order (the parallel schedule is nondeterministic).
        loaded.sort_by(|left, right| left.0.cmp(&right.0));

        let mut transformed = 0;
        {
            for (path, result) in loaded {
                if self
                    .graph
                    .indices
                    .get(path.as_ref())
                    .is_some_and(|index| self.graph.modules[*index].is_some())
                {
                    continue;
                }
                let loaded = result?;
                diagnostics.extend(loaded.diagnostics);
                transformed += 1;
                let source = self.graph.intern(path.clone());
                let mut dependencies = Vec::with_capacity(loaded.dependencies.len());
                for (specifier, target, demand) in loaded.dependencies {
                    let target_index = self.graph.intern(target.clone());
                    if record_delta {
                        delta
                            .edge_updates
                            .push(((path.to_string(), target.to_string()), 1));
                    }
                    dependencies.push((specifier, target_index, demand));
                }
                self.graph.modules[source] = Some(ModuleState {
                    hash: loaded.hash,
                    code_hash: loaded.code_hash,
                    dependencies,
                    pruned_imports: loaded.pruned_imports,
                    source: loaded.source,
                    flat_module: loaded.flat_module,
                    code: loaded.code,
                    assets: loaded.assets,
                    provider_assets: loaded.provider_assets,
                    css: loaded.css,
                    css_source_files: loaded.css_source_files,
                    css_external_imports: loaded.css_external_imports,
                    externals: loaded.externals,
                    droppable: loaded.droppable,
                    liveness: loaded.liveness,
                    uses_top_level_await: loaded.uses_top_level_await,
                    uses_cjs_globals: loaded.uses_cjs_globals,
                    uses_dirname: loaded.uses_dirname,
                    workers: loaded.workers,
                    map: loaded.map,
                });
            }
        }
        Ok(transformed)
    }

    fn load_module(
        &mut self,
        path: &Path,
        diagnostics: &mut Vec<Diagnostic>,
    ) -> Result<ModuleState, String> {
        let id = module_id(path);
        // A build-generated virtual module (its source is not on disk) claims this
        // id first.
        if let Some(source) = self.resolution_cache.virtual_module_source(&id) {
            let special =
                diffpack_default_loader::module::virtual_module(source, |path, source| {
                    compile_synthetic_with(self.compiler.as_ref(), path, source, Target::Server)
                });
            let resolved = resolve_special_dependencies(
                &self.resolver,
                &self.resolution_cache,
                self.compiler.as_ref(),
                &id,
                self.target,
                &special,
                diagnostics,
            );
            let dependencies = resolved
                .dependencies
                .into_iter()
                .map(|(specifier, target, demand)| (specifier, self.graph.intern(target), demand))
                .collect();
            return Ok(ModuleState {
                hash: special.hash,
                code_hash: special.hash,
                dependencies,
                pruned_imports: resolved.pruned_imports,
                source: id.clone(),
                flat_module: special.flat_module,
                code: special.code,
                assets: special.assets,
                provider_assets: Vec::new(),
                css: special.css,
                css_source_files: special.css_source_files,
                css_external_imports: special.css_external_imports,
                externals: resolved.externals,
                droppable: false,
                liveness: ModuleLiveness::default(),
                uses_top_level_await: false,
                uses_cjs_globals: false,
                uses_dirname: false,
                workers: Vec::new(),
                map: None,
            });
        }
        // A loader (query, stylesheet, or asset) may claim this id before it is
        // ever read as JavaScript.
        if let Some(special) = load_special_module(
            &id,
            path,
            self.target,
            self.hmr,
            &self.resolution_cache,
            self.compiler.as_ref(),
            self.special_modules.as_ref(),
        ) {
            let special = special?;
            let resolved = resolve_special_dependencies(
                &self.resolver,
                &self.resolution_cache,
                self.compiler.as_ref(),
                &id,
                self.target,
                &special,
                diagnostics,
            );
            let dependencies = resolved
                .dependencies
                .into_iter()
                .map(|(specifier, target, demand)| (specifier, self.graph.intern(target), demand))
                .collect();
            // A `?worker` module bundles its referenced entry as a self-contained
            // worker chunk. The key matches the `__diffpack_worker__<key>__`
            // placeholder the synthesizer emitted, so the emit-step substitution
            // (shared with the `new Worker(new URL(...))` path) resolves it to the
            // emitted URL.
            let resource = ResourceId::parse(&id);
            let workers = if loader_policy::kind(&resource) == Some(LoaderKind::Worker) {
                vec![(
                    diffpack_default_loader::asset::worker_key(Path::new(&resource.path)),
                    PathBuf::from(&resource.path),
                )]
            } else {
                Vec::new()
            };
            return Ok(ModuleState {
                hash: special.hash,
                code_hash: special.hash,
                dependencies,
                pruned_imports: resolved.pruned_imports,
                source: id.clone(),
                flat_module: special.flat_module,
                code: special.code,
                assets: special.assets,
                provider_assets: Vec::new(),
                css: special.css,
                css_source_files: special.css_source_files,
                css_external_imports: special.css_external_imports,
                externals: resolved.externals,
                droppable: false,
                liveness: ModuleLiveness::default(),
                uses_top_level_await: false,
                uses_cjs_globals: false,
                uses_dirname: false,
                workers,
                map: None,
            });
        }
        let read_started = frontend_profile::start();
        let provided = self.resolution_cache.provider_source(&id, self.target)?;
        let externally_loaded = provided.is_some();
        let source = match &provided {
            Some((source, _, _, _, _)) => source.clone(),
            None => fs::read_to_string(path)
                .map_err(|error| format!("cannot read {}: {error}", path.display()))?,
        };
        frontend_profile::finish(Phase::Read, read_started);
        let hash = content_hash(source.as_bytes());
        if let Some(current) = self
            .graph
            .indices
            .get(id.as_ref())
            .and_then(|index| self.graph.modules[*index].as_ref())
            && current.hash == hash
        {
            return Ok(current.clone());
        }
        // A `.vue`/`.svelte` component is compiled to JavaScript by the app's own
        // compiler FIRST; everything below then treats the result as the module's
        // source, so the component's imports become graph edges and its
        // TypeScript is stripped by the ordinary transform.
        let (component_code, language, mut component, mut provider_assets) =
            if let Some((_, language, assets, watch_files, provider_diagnostics)) = provided {
                diagnostics.extend(provider_diagnostics);
                let mut side_effects = ComponentSideEffects::default();
                side_effects.css_source_files = watch_files;
                (None, language, side_effects, assets)
            } else {
                match diffpack_default_loader::module::precompile_component(
                    path,
                    &source,
                    self.resolution_cache.css_preprocess.root_path(),
                    self.resolution_cache.css_preprocess.postcss.as_deref(),
                )? {
                    Some(compiled) => (
                        Some(compiled.code),
                        compiled.language,
                        compiled.side_effects,
                        Vec::new(),
                    ),
                    None => (
                        None,
                        diffpack_core::transform::SourceLanguage::FromPath,
                        ComponentSideEffects::default(),
                        Vec::new(),
                    ),
                }
            };
        let module_text = component_code.as_deref().unwrap_or(source.as_str());
        let source =
            self.resolution_cache
                .apply_vite_replacements(path, module_text, self.target)?;
        let source_was_rewritten = matches!(source, Cow::Owned(_));
        let (source, language, provider_rewritten) = if externally_loaded {
            (source.into_owned(), language, false)
        } else {
            let before_provider = source.into_owned();
            let (source, language, assets, watch_files, provider_diagnostics) = self
                .resolution_cache
                .transform_external_source(&id, &before_provider, language, self.target)?;
            diagnostics.extend(provider_diagnostics);
            let rewritten = source != before_provider;
            provider_assets.extend(assets);
            component.css_source_files.extend(watch_files);
            (source, language, rewritten)
        };
        let project_config = diffpack_default_loader::jsx_project_config::project_config(
            &self.resolver,
            path,
            &self.resolution_cache.jsx,
            self.compiler.is_generated_path(path),
        )?;
        let mut transformed = self.compiler.compile(CompileRequest {
            path,
            source: &source,
            target: self.target,
            hmr: self.hmr,
            refresh: self.hmr && self.target == Target::Client,
            jsx: self.resolution_cache.jsx_extensions,
            project_config: &project_config,
            language,
            source_maps: self.config.source_maps,
        });
        // The text the map's positions were measured against is `source`, which
        // is the file on disk ONLY when neither a component compiler nor Vite's
        // replacements rewrote it. When one did, the map says so, so nothing can
        // read its positions as offsets into the file.
        diffpack_default_loader::module::mark_rewritten_source(
            &mut transformed,
            component_code.is_some(),
            source_was_rewritten || provider_rewritten,
        );
        diagnostics.extend(source_diagnostics(path, &transformed.diagnostics));

        // A component's `<style>` `@import`s are graph edges of the component
        // module, exactly as a stylesheet's own `@import`s are of that stylesheet.
        // Borrowed (no copy) for every ordinary module, which is all of them but
        // the components.
        let (dependency_specifiers, dependency_demands) = component.dependencies(&transformed);
        let resolved_dependencies = resolve_dependencies(
            &self.resolver,
            &self.resolution_cache,
            self.compiler.as_ref(),
            path,
            self.target,
            &dependency_specifiers,
            &dependency_demands,
            diagnostics,
        );
        let dependencies = resolved_dependencies
            .dependencies
            .into_iter()
            .map(|(specifier, target, demand)| (specifier, self.graph.intern(target), demand))
            .collect();

        let code_hash = content_hash(transformed.code.as_bytes());
        let droppable =
            diffpack_default_loader::side_effects::droppable_with_diagnostics(path, diagnostics);
        Ok(ModuleState {
            hash,
            code_hash,
            dependencies,
            pruned_imports: resolved_dependencies.pruned_imports,
            source: Arc::from(source),
            flat_module: transformed.flat_module,
            code: transformed.code,
            assets: component.assets,
            provider_assets,
            css: component.css,
            css_source_files: component.css_source_files,
            css_external_imports: component.css_external_imports,
            externals: resolved_dependencies.externals,
            droppable,
            liveness: transformed.liveness,
            uses_top_level_await: transformed.uses_top_level_await,
            uses_cjs_globals: transformed.uses_cjs_globals,
            uses_dirname: transformed.uses_dirname,
            workers: resolve_worker_entries(&self.resolver, path, &transformed.workers)?,
            map: transformed.map,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn render_best(
        &self,
        reachable: &[DenseModuleId],
        roots: &[DenseModuleId],
        chunk_names: &HashMap<DenseModuleId, String>,
        chunk_files: &[String],
        runtime_ids: &[Option<usize>],
        global_demands: &[ExportDemand],
        prerequisites: &[String],
        is_main: bool,
        flat_allowed: bool,
        async_modules: &AsyncModules,
        format: ModuleFormat,
        hmr: bool,
        cancel: EmitCancel<'_>,
    ) -> Result<Option<RenderedBundle>, String> {
        // The flat path emits a plain concatenation with no per-module factory
        // registry, so it has no place to install `module.hot`; a dev (hmr) build
        // always renders through the registry runtime so every module is HMR-capable.
        // `flat_allowed` is the caller's build-wide verdict on whether the chunk
        // partition is closed enough for scope hoisting at all (see `chunk_plan`).
        if flat_allowed
            && !hmr
            && !async_modules.any
            && !(format == ModuleFormat::Cjs
                && reachable.iter().any(|dense| {
                    self.graph.modules[*dense]
                        .as_ref()
                        .is_some_and(|module| module.code.contains("import.meta"))
                }))
            && let Some(flat) = self.render_flat(
                reachable,
                roots,
                chunk_names,
                global_demands,
                is_main,
                format,
            )
        {
            return Ok(Some(flat));
        }
        // A top-level `await` that reaches the registry runtime is rendered as an
        // `async` factory (see `render_runtime`); `async_module_closure` has already
        // refused every import site that cannot carry the matching `await`.
        self.render_runtime(
            reachable,
            roots,
            chunk_names,
            chunk_files,
            runtime_ids,
            global_demands,
            prerequisites,
            is_main,
            async_modules,
            format,
            hmr,
            cancel,
        )
    }

    fn render_flat(
        &self,
        reachable: &[DenseModuleId],
        roots: &[DenseModuleId],
        chunk_names: &HashMap<DenseModuleId, String>,
        global_demands: &[ExportDemand],
        is_main: bool,
        format: ModuleFormat,
    ) -> Option<RenderedBundle> {
        let modules = self
            .graph
            .modules
            .iter()
            .map(|module| {
                let module = module.as_ref()?;
                Some(FlatRenderModule {
                    flat: module.flat_module.as_ref()?,
                    dependencies: &module.dependencies,
                    pruned_imports: &module.pruned_imports,
                    map: module.map.as_ref(),
                    has_externals: !module.externals.is_empty(),
                    uses_cjs_globals: module.uses_cjs_globals,
                })
            })
            .collect::<Vec<_>>();
        let prelude = (is_main && format == ModuleFormat::BrowserEsm)
            .then(|| {
                self.runtime_policy
                    .flat_entry_prelude(self.config.browser_process_shim)
            })
            .flatten();
        self.frontend_pool.install(|| {
            render_flat_chunk(
                &self.graph.ids,
                &modules,
                self.graph.entry,
                reachable,
                roots,
                chunk_names,
                global_demands,
                is_main,
                format,
                prelude.as_deref(),
            )
        })
    }

    fn static_execution_order(
        &self,
        root: DenseModuleId,
        allowed: &HashSet<DenseModuleId>,
    ) -> Option<Vec<DenseModuleId>> {
        static_execution_order(&StaticModuleGraphView { graph: &self.graph }, root, allowed)
    }

    #[allow(clippy::too_many_arguments)]
    fn render_runtime(
        &self,
        reachable: &[DenseModuleId],
        roots: &[DenseModuleId],
        chunk_names: &HashMap<DenseModuleId, String>,
        chunk_files: &[String],
        runtime_ids: &[Option<usize>],
        global_demands: &[ExportDemand],
        prerequisites: &[String],
        is_main: bool,
        async_modules: &AsyncModules,
        format: ModuleFormat,
        hmr: bool,
        cancel: EmitCancel<'_>,
    ) -> Result<Option<RenderedBundle>, String> {
        let modules = self
            .graph
            .modules
            .iter()
            .enumerate()
            .map(|(dense, module)| {
                let module = module.as_ref()?;
                Some(RuntimeRenderModule {
                    id: &self.graph.ids[dense],
                    code: &module.code,
                    dependencies: &module.dependencies,
                    pruned_imports: &module.pruned_imports,
                    map: module.map.as_ref(),
                    uses_dirname: module.uses_dirname,
                })
            })
            .collect::<Vec<_>>();
        let Some(fragments) = self.frontend_pool.install(|| {
            render_runtime_fragments(
                &modules,
                reachable,
                roots,
                chunk_names,
                runtime_ids,
                global_demands,
                async_modules,
                format,
                &|| cancel.cancelled(),
            )
        }) else {
            return Ok(None);
        };
        let policy_modules = reachable
            .iter()
            .filter_map(|&dense| {
                self.graph.modules[dense]
                    .as_ref()
                    .map(|module| RuntimePolicyModule {
                        id: self.graph.ids[dense].as_ref(),
                        source: module.source.as_ref(),
                    })
            })
            .collect::<Vec<_>>();
        let runtime_policy = self.runtime_policy.configure(RuntimePolicyRequest {
            format,
            is_main,
            hmr,
            entry_id: self.graph.ids[self.graph.entry].as_ref(),
            entry_runtime_id: runtime_ids[self.graph.entry]
                .expect("the entry module must have a deterministic runtime ID"),
            any_async: async_modules.any,
            base: &self.config.base,
            chunk_files,
            modules: &policy_modules,
            browser_process_shim: self.config.browser_process_shim,
        })?;
        let node_host_prelude = (format == ModuleFormat::Esm)
            .then(|| diffpack_default_loader::runtime::node_esm_chunk_prelude(is_main));
        let mut preludes = Vec::new();
        if let Some(prelude) = node_host_prelude.as_deref() {
            preludes.push(prelude);
        }
        for prelude in &runtime_policy.entry_preludes {
            preludes.push(prelude.value.as_str());
        }
        if let Some(prelude) = runtime_policy.compatibility_prelude.as_ref() {
            preludes.push(prelude.value.as_str());
        }
        let header = render_runtime_header(format, prerequisites, &preludes);
        let prelude = header.prelude;
        let prerequisite_loads = header.prerequisite_loads;
        let header_lines = header.generated_lines;
        let literals = assemble_runtime_literals(fragments, header_lines, |dense| {
            self.graph.modules[dense]
                .as_ref()
                .and_then(|module| module.map.as_ref())
        });
        let modules = literals.modules;
        let maps = literals.import_maps;
        let chunks = literals.chunk_maps;
        let mappings = literals.mappings;

        let require_native = match format {
            ModuleFormat::Esm => {
                "const requireNative=__diffpackCreateRequire(import.meta.url);".to_string()
            }
            ModuleFormat::BrowserEsm => runtime_policy
                .browser_require_native
                .as_ref()
                .map(|capability| capability.value.clone())
                .expect("browser runtime policy must provide requireNative"),
            ModuleFormat::Cjs => {
                r#"const requireNative=typeof require==="function"?require:null;"#.to_string()
            }
        };
        let hot_policy = runtime_policy
            .hot
            .as_ref()
            .map(|policy| policy.value.borrowed());
        Ok(Some(render_registry_runtime(
            self.graph.ids[self.graph.entry].as_ref(),
            self.graph.entry,
            roots,
            runtime_ids,
            async_modules,
            is_main,
            format,
            prelude,
            prerequisite_loads,
            modules,
            maps,
            chunks,
            mappings,
            require_native,
            hot_policy,
        )))
    }

    /// The source map for a READABLE (un-minified) chunk.
    ///
    /// Every token comes from the Oxc printer: the position it actually wrote in
    /// the module's lowered text, moved onto the chunk's line by the render, and
    /// the span of the AST node it printed. Nothing is inferred from a line
    /// number.
    ///
    /// A generated line with no token — the runtime wrapper, the export footer,
    /// the browser prelude, the CJS `exports=module.exports=...` preamble, the
    /// factory headers, the registry literals, and any line a render rewrite could
    /// not account for — is explicitly marked UNMAPPED, with a one-field segment
    /// at its column 0. That marker is the whole mechanism, not a formality:
    /// OMITTING a token does not make a position unmapped. Every consumer (Node's
    /// `--enable-source-maps`, which is how `diffpack start` runs the server, and
    /// DevTools) resolves a position to the last mapping at or before it in the
    /// WHOLE file, ignoring line boundaries, so a line with no segments silently
    /// inherits the previous line's origin — which is how a frame inside the
    /// bundler's own `__require` came out named after an author identifier in an
    /// author file. A one-field segment is the format's way of saying "from here
    /// on, nothing", and it is what stops the bleed.
    ///
    /// Sources are project-relative `diffpack://` labels (never an absolute path
    /// leak or a `..` traversal) whose inlined `sourcesContent` is the exact text
    /// the positions were measured against, so a rewritten source is labelled as
    /// one and cannot be read as the file on disk.
    fn source_map(&self, mappings: &[ModuleMapping], output_name: &str, code: &str) -> String {
        let root = self.map_source_root();
        let labels = mappings
            .iter()
            .map(|mapping| {
                let origin = self.graph.modules[mapping.dense_index]
                    .as_ref()
                    .and_then(|module| module.map.as_ref())
                    .map_or(MapOrigin::File, ModuleSourceMap::origin);
                (
                    mapping.dense_index,
                    self.source_label(mapping.dense_index, root, origin),
                )
            })
            .collect::<HashMap<_, _>>();
        serialize_readable_source_map(
            &ModuleMapView { graph: &self.graph },
            &labels,
            mappings,
            output_name,
            code,
        )
    }

    /// Composes the two maps a minified chunk can honestly produce into one that
    /// resolves a MINIFIED position back to the correct ORIGINAL source position:
    ///
    /// - `readable_mappings` — the REAL readable-generated -> original tokens the
    ///   render produced for each module region (see [`ModuleMapping::tokens`]);
    /// - `minified_map` — minified position -> readable-generated position,
    ///   emitted by Oxc codegen when it re-prints the readable chunk minified.
    ///
    /// For each token of the minified map, its readable position is resolved
    /// against the readable tokens: the last readable token at or before it on the
    /// SAME line is the construct the minifier was printing, and that token's
    /// original file/line/column (and name) is what the minified position came
    /// from.
    ///
    /// A minified position with no readable token before it on its line resolved
    /// into no module at all — a synthetic bundler region, or text a render rewrite
    /// could not account for — and is written out as an explicit UNMAPPED
    /// (one-field) segment, never given a fabricated origin and never merely
    /// omitted. Omitting it would not mark it unmapped: a consumer resolves a
    /// position to the last mapping at or before it in the whole file, so a skipped
    /// token silently inherits the mapping of whatever author code the minifier
    /// printed before it. Minified output is one long line of interleaved author
    /// code and bundler runtime, so this is the only thing keeping a frame inside
    /// `__require` from being reported as a line of somebody's component.
    ///
    /// A chunk whose minified map resolves into no original module at all is a
    /// hard error naming the chunk, never a silently empty map.
    fn compose_source_map(
        &self,
        readable_mappings: &[ModuleMapping],
        minified_map: &oxc_sourcemap::SourceMap,
        output_name: &str,
        chunk_name: &str,
    ) -> Result<String, String> {
        let root = self.map_source_root();
        // Labels are owned here so they outlive the borrowing builder.
        let labels: HashMap<DenseModuleId, String> = readable_mappings
            .iter()
            .map(|mapping| {
                let origin = self.graph.modules[mapping.dense_index]
                    .as_ref()
                    .and_then(|module| module.map.as_ref())
                    .map_or(MapOrigin::File, ModuleSourceMap::origin);
                (
                    mapping.dense_index,
                    self.source_label(mapping.dense_index, root, origin),
                )
            })
            .collect();
        // Every readable token in the chunk, sorted by generated position, each
        // remembering which module it belongs to.
        let mut readable: Vec<(MapToken, DenseModuleId)> = readable_mappings
            .iter()
            .flat_map(|mapping| {
                mapping
                    .tokens
                    .iter()
                    .map(move |token| (*token, mapping.dense_index))
            })
            .collect();
        readable.sort_by_key(|(token, _)| (token.generated_line, token.generated_column));

        // Resolution (the binary search per minified token, plus the occasional
        // name verification against the module's source text) is read-only work
        // over millions of tokens for a large chunk, so it runs across the pool in
        // generated-order slices. Emission stays serial below: source ids and
        // names must be assigned in first-use order for the output to stay
        // byte-identical to the single-threaded composition.
        let tokens: Vec<oxc_sourcemap::Token> = minified_map.get_tokens().collect();
        let module_maps = ModuleMapView { graph: &self.graph };
        const PARALLEL_TOKEN_SLICE: usize = 1 << 15;
        let resolved: Vec<Option<ResolvedMinifiedToken<'_>>> = if tokens.len()
            > PARALLEL_TOKEN_SLICE
        {
            self.frontend_pool.install(|| {
                tokens
                    .par_chunks(PARALLEL_TOKEN_SLICE)
                    .flat_map_iter(|slice| {
                        // Byte offsets of each line of a module's source, built
                        // lazily the first time a name has to be checked against
                        // the source text. Per-slice: tokens are in generated
                        // order, so a slice's tokens cluster in few modules.
                        let mut source_lines: HashMap<DenseModuleId, Vec<usize>> = HashMap::new();
                        let mut hint = 0usize;
                        slice
                            .iter()
                            .map(|minified| {
                                resolve_minified_token(
                                    &module_maps,
                                    minified,
                                    minified_map,
                                    &readable,
                                    &mut hint,
                                    &mut source_lines,
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect()
            })
        } else {
            let mut source_lines: HashMap<DenseModuleId, Vec<usize>> = HashMap::new();
            let mut hint = 0usize;
            tokens
                .iter()
                .map(|minified| {
                    resolve_minified_token(
                        &module_maps,
                        minified,
                        minified_map,
                        &readable,
                        &mut hint,
                        &mut source_lines,
                    )
                })
                .collect()
        };
        serialize_composed_source_map(
            &module_maps,
            &labels,
            readable_mappings,
            &tokens,
            resolved,
            output_name,
            chunk_name,
        )
    }

    /// The directory every emitted `sources` label is relative to: the PROJECT
    /// root, computed once for the whole build.
    ///
    /// It has to be one directory for the whole build, not per map. A per-chunk
    /// common ancestor collapses to the module's own directory whenever a chunk
    /// holds one module — and then the label is a bare file name, so cal.com
    /// emitted `diffpack:///Setup.tsx` for nine genuinely different `Setup.tsx`
    /// files and `diffpack:///add.ts` for thirty different `add.ts` files. A source
    /// URL is an IDENTITY: DevTools' source tree and every error reporter dedupe on
    /// it, so two files sharing one URL means one file's content is shown for the
    /// other's frames.
    ///
    /// The root is the OUTERMOST ancestor of the entry that holds a `package.json`
    /// — the workspace root of a monorepo, so a sibling package keeps its
    /// `packages/ui/...` path instead of being pushed outside. Two directories can
    /// never be the root: the filesystem root and the user's home (or any ancestor
    /// of it), because a label relative to either is an absolute path in all but
    /// name and publishes the user's directory layout. With neither a package root
    /// nor a usable common ancestor available, the answer is `None` — no shared
    /// root exists, and every module is labelled by
    /// [`Self::external_source_label`] instead of by a path that would leak.
    fn map_source_root(&self) -> Option<&Path> {
        self.map_root
            .get_or_init(|| {
                let home = std::env::var_os("HOME").map(PathBuf::from);
                let usable = |directory: &Path| {
                    directory.parent().is_some()
                        && home
                            .as_deref()
                            .is_none_or(|home| !home.starts_with(directory))
                };
                let entry = PathBuf::from(
                    ResourceId::parse(self.graph.ids[self.graph.entry].as_ref()).path,
                );
                let mut outermost: Option<PathBuf> = None;
                let mut cursor = entry.parent();
                while let Some(directory) = cursor {
                    if usable(directory) && directory.join("package.json").is_file() {
                        outermost = Some(directory.to_path_buf());
                    }
                    cursor = directory.parent();
                }
                if outermost.is_some() {
                    return outermost;
                }
                // No package root anywhere above the entry (a bare script, a test
                // fixture): the common ancestor of every module in the build is
                // still a real shared directory, and using it keeps each module's
                // path inside the build.
                let mut common: Option<PathBuf> = None;
                for id in &self.graph.ids {
                    let path = PathBuf::from(ResourceId::parse(id.as_ref()).path);
                    if !path.is_absolute() {
                        continue;
                    }
                    let directory = path
                        .parent()
                        .map(Path::to_path_buf)
                        .unwrap_or_else(|| path.clone());
                    common = Some(match common {
                        None => directory,
                        Some(existing) => common_ancestor(&existing, &directory),
                    });
                }
                common.filter(|directory| usable(directory))
            })
            .as_deref()
    }

    /// A label for a module that is NOT under the build's source root: a package
    /// in a store outside the project, a symlinked workspace package, a file on
    /// another volume, or any module at all when the build has no shared root.
    ///
    /// Its absolute path must never reach the emitted map — that names the machine
    /// and the user, and production maps are served to browsers — but its bare file
    /// name is not enough either, because two different `index.js` files would
    /// become one source. So the label keeps the part of the path that identifies
    /// the file WITHIN its own package (from the nearest directory holding a
    /// `package.json`, that directory included) and disambiguates it with a short
    /// digest of the full path, which is stable across chunks and across rebuilds
    /// and reveals nothing about where the file lives.
    fn external_source_label(path: &Path) -> String {
        let mut anchor = None;
        let mut cursor = path.parent();
        while let Some(directory) = cursor {
            if directory.join("package.json").is_file() {
                anchor = directory.parent();
                break;
            }
            cursor = directory.parent();
        }
        let within = anchor
            .and_then(|anchor| path.strip_prefix(anchor).ok())
            .or_else(|| path.file_name().map(Path::new))
            .unwrap_or(path);
        let mut hasher = DefaultHasher::new();
        path.hash(&mut hasher);
        let digest = hasher.finish();
        let tail = within
            .components()
            .filter_map(|component| component.as_os_str().to_str())
            .collect::<Vec<_>>()
            .join("/");
        format!("external/{digest:016x}/{tail}")
    }

    /// A stable, non-leaking `sources` label for a module. Emitted map paths must
    /// be project-relative: never an absolute filesystem path (a privacy leak) and
    /// never a `..` traversal. The module's on-disk path is made relative to `root`
    /// (see [`Self::map_source_root`]) and served under a `diffpack://` scheme so
    /// DevTools shows the real project-relative source without exposing where the
    /// project lives on disk. A module outside `root` is labelled by
    /// [`Self::external_source_label`], which is likewise absolute-free and
    /// traversal-free but still unique. A virtual/plugin module has no on-disk path
    /// at all and keeps its own id. Any query/fragment is preserved so distinct
    /// graph keys (`app.css` vs `app.css?url`) stay distinct sources.
    fn source_label(&self, dense: DenseModuleId, root: Option<&Path>, origin: MapOrigin) -> String {
        let resource = ResourceId::parse(self.graph.ids[dense].as_ref());
        let path = PathBuf::from(&resource.path);
        let mut label = if path.is_absolute() {
            let relative = root
                .and_then(|root| path.strip_prefix(root).ok())
                .filter(|relative| {
                    relative.components().all(|component| {
                        !matches!(
                            component,
                            Component::ParentDir | Component::RootDir | Component::Prefix(_)
                        )
                    })
                });
            match relative {
                Some(relative) => relative
                    .components()
                    .filter_map(|component| component.as_os_str().to_str())
                    .collect::<Vec<_>>()
                    .join("/"),
                None => Self::external_source_label(&path),
            }
        } else {
            // A virtual module id (`tanstack-start-manifest:v`, a plugin's own
            // namespace): not a path at all, so there is nothing to make relative
            // and nothing on disk to leak.
            resource.path.replace('\\', "/")
        };
        if label.is_empty() {
            label = "module".to_string();
        }
        fn append(label: &mut String, separator: &mut char, key: &str, value: &str) {
            label.push(*separator);
            *separator = '&';
            label.push_str(key);
            if !value.is_empty() {
                label.push('=');
                label.push_str(value);
            }
        }
        let mut separator = '?';
        if let Some(query) = &resource.query {
            append(&mut label, &mut separator, query, "");
        }
        // A source diffpack GENERATED from this file (an MDX compile, an RSC
        // directive rewrite, a route split, the `next/font` macro, a component
        // compiler, a Vite replacement) is labelled as such. The positions in the
        // map index that generated text — which is what the map inlines as this
        // source's content — so labelling it with the bare filename would invite
        // a reader to line the numbers up against the file on disk.
        //
        // The GRAPH is part of that identity. The same file rewritten for the
        // browser and for the server is two different texts: `import.meta.env` and
        // the `define` substitutions resolve differently, and the dead branches
        // that fall out of them differ with them. cal.com ships one such file
        // (next-i18next's `createConfig.js`, 8,270 bytes on the client and 13,430
        // on the server), and under one shared URL a consumer that dedupes by
        // source URL shows one graph's text for the other graph's frames.
        if let MapOrigin::Generated(stage) = origin {
            append(&mut label, &mut separator, "diffpack-generated", stage);
            append(
                &mut label,
                &mut separator,
                "diffpack-graph",
                match self.target {
                    Target::Client => "client",
                    Target::Server => "server",
                    Target::IsolatedServer => "react-server",
                },
            );
        }
        if let Some(fragment) = &resource.fragment {
            label.push('#');
            label.push_str(fragment);
        }
        format!("diffpack:///{label}")
    }

    /// Aggregates, for every module, the union of export demand placed on it by
    /// all consumers in `sources`. An emitted module keeps only the exports its
    /// consumers actually ask for, so this must be computed over the FULL set of
    /// reachable modules — not a single chunk's closure. A module and one of its
    /// consumers frequently land in different chunks (e.g. a shared package index
    /// consumed by a route split), and a chunk-local demand would wrongly shake
    /// away exports the other chunk imports at runtime.
    /// Computes the export-level LIVE subset of a module-level reachable set:
    /// generic, `sideEffects`-aware dead-module elimination that matches
    /// Rollup/esbuild semantics.
    ///
    /// A reachable module is live when it is the entry, a dynamic-import chunk
    /// root reached from live code, a module whose `package.json` does NOT
    /// authorize dropping it (so its side effects must run whenever a live module
    /// imports it), or a module at least one of whose exports is used — directly
    /// or transitively through re-export barrels — by another live module. The
    /// pass iterates to a fixpoint; dropping a module can make its own
    /// dependencies unused, which the worklist re-propagates.
    ///
    /// The distinction that makes barrel tree-shaking work is body use vs
    /// re-export: an imported binding referenced in real module code
    /// ([`ModuleLiveness::body_uses`]) places demand on its source
    /// unconditionally once the module runs, whereas a binding merely forwarded
    /// as one of this module's exports ([`ModuleLiveness::reexports`]) places
    /// demand only when that export is itself used. A `sideEffects:false` module
    /// reached ONLY through a barrel whose forwarded binding no live module uses
    /// therefore receives no demand and is dropped.
    ///
    /// The result is a deterministic function of the graph, independent of
    /// worklist order (all state grows monotonically), so a full build and an
    /// incremental build of the same graph drop exactly the same modules — the
    /// output stays byte-identical. This pass reads, and never mutates, the
    /// incremental reachability index.
    pub fn live_modules(&self, reachable: &BTreeSet<ModuleId>) -> BTreeSet<ModuleId> {
        let reachable_dense = reachable
            .iter()
            .filter_map(|id| self.graph.indices.get(id.as_str()).copied())
            .filter(|&module| self.graph.modules[module].is_some())
            .collect::<HashSet<_>>();
        derive_live_modules(
            &LinkModuleGraphView { graph: &self.graph },
            &reachable_dense,
        )
        .into_iter()
        .map(|module| self.graph.ids[module].to_string())
        .collect()
    }

    #[doc(hidden)]
    pub fn export_demands(&self, sources: &[DenseModuleId]) -> Vec<ExportDemand> {
        derive_export_demands(&LinkModuleGraphView { graph: &self.graph }, sources)
    }
}

/// Folds an optional runtime id into a chunk render key, distinguishing `None`
/// from `Some(0)`.
struct RenderKeyView<'a> {
    modules: &'a [Option<ModuleState>],
    runtime_ids: &'a [Option<usize>],
    demands: &'a [ExportDemand],
    async_modules: &'a AsyncModules,
}

fn dependency_key_view<'a>(
    dependency: &'a (String, DenseModuleId, DependencyDemand),
) -> RenderKeyDependency<'a> {
    RenderKeyDependency {
        specifier: &dependency.0,
        target: dependency.1,
        dynamic: dependency.2.dynamic,
        eager: dependency.2.eager,
        all: dependency.2.all,
        names: &dependency.2.names,
    }
}

impl RenderKeyGraph for RenderKeyView<'_> {
    type ExportNames<'a>
        = std::iter::Map<std::collections::hash_set::Iter<'a, String>, fn(&'a String) -> &'a str>
    where
        Self: 'a;
    type Dependencies<'a>
        = std::iter::Map<
        std::slice::Iter<'a, (String, DenseModuleId, DependencyDemand)>,
        fn(&'a (String, DenseModuleId, DependencyDemand)) -> RenderKeyDependency<'a>,
    >
    where
        Self: 'a;

    fn code_hash(&self, module: DenseModuleId) -> Option<u64> {
        self.modules
            .get(module)?
            .as_ref()
            .map(|module| module.code_hash)
    }

    fn runtime_id(&self, module: DenseModuleId) -> Option<usize> {
        self.runtime_ids.get(module).copied().flatten()
    }

    fn export_all(&self, module: DenseModuleId) -> bool {
        self.demands.get(module).is_some_and(|demand| demand.all)
    }

    fn export_names(&self, module: DenseModuleId) -> Self::ExportNames<'_> {
        fn as_str(value: &String) -> &str {
            value
        }
        self.demands[module].names.iter().map(as_str)
    }

    fn is_async(&self, module: DenseModuleId) -> bool {
        self.async_modules.is_async(module)
    }

    fn dependencies(&self, module: DenseModuleId) -> Self::Dependencies<'_> {
        self.modules[module]
            .as_ref()
            .expect("render-key member exists")
            .dependencies
            .iter()
            .map(dependency_key_view)
    }
}

fn load_uncached(
    resolver: &Resolvers,
    resolution_cache: &ResolutionCache,
    compiler: &dyn ModuleCompiler,
    special_modules: &dyn SpecialModulePolicy,
    path: &Path,
    target: Target,
    hmr: bool,
    source_maps: bool,
) -> Result<LoadedModule, String> {
    let id = path.to_string_lossy();
    // A build-generated virtual module (its source is not on disk) claims this id
    // first.
    if let Some(source) = resolution_cache.virtual_module_source(&id) {
        let special = diffpack_default_loader::module::virtual_module(source, |path, source| {
            compile_synthetic_with(compiler, path, source, Target::Server)
        });
        let mut diagnostics = Vec::new();
        let resolved = resolve_special_dependencies(
            resolver,
            resolution_cache,
            compiler,
            &id,
            target,
            &special,
            &mut diagnostics,
        );
        return Ok(LoadedModule {
            hash: special.hash,
            code_hash: special.hash,
            dependencies: resolved.dependencies,
            pruned_imports: resolved.pruned_imports,
            source: Arc::from(id.as_ref()),
            flat_module: special.flat_module,
            code: special.code,
            diagnostics,
            assets: special.assets,
            provider_assets: Vec::new(),
            css: special.css,
            css_source_files: special.css_source_files,
            css_external_imports: special.css_external_imports,
            externals: resolved.externals,
            droppable: false,
            liveness: ModuleLiveness::default(),
            uses_top_level_await: false,
            uses_cjs_globals: false,
            uses_dirname: false,
            workers: Vec::new(),
            map: None,
        });
    }
    // A loader (query, stylesheet, or asset) may claim this id before it is ever
    // read as JavaScript.
    if let Some(special) = load_special_module(
        &id,
        path,
        target,
        hmr,
        resolution_cache,
        compiler,
        special_modules,
    ) {
        let special = special?;
        let mut diagnostics = Vec::new();
        let resolved = resolve_special_dependencies(
            resolver,
            resolution_cache,
            compiler,
            &id,
            target,
            &special,
            &mut diagnostics,
        );
        // A `?worker` module registers its referenced entry as a worker chunk;
        // the key matches the `__diffpack_worker__<key>__` placeholder in the
        // synthesized constructor so the emit-step substitution resolves it.
        let resource = ResourceId::parse(&id);
        let workers = if loader_policy::kind(&resource) == Some(LoaderKind::Worker) {
            vec![(
                diffpack_default_loader::asset::worker_key(Path::new(&resource.path)),
                PathBuf::from(&resource.path),
            )]
        } else {
            Vec::new()
        };
        return Ok(LoadedModule {
            hash: special.hash,
            code_hash: special.hash,
            dependencies: resolved.dependencies,
            pruned_imports: resolved.pruned_imports,
            source: Arc::from(id.as_ref()),
            flat_module: special.flat_module,
            code: special.code,
            diagnostics,
            assets: special.assets,
            provider_assets: Vec::new(),
            css: special.css,
            css_source_files: special.css_source_files,
            css_external_imports: special.css_external_imports,
            externals: resolved.externals,
            droppable: false,
            liveness: ModuleLiveness::default(),
            uses_top_level_await: false,
            uses_cjs_globals: false,
            uses_dirname: false,
            workers,
            map: None,
        });
    }
    let mut diagnostics = Vec::new();
    let read_started = frontend_profile::start();
    let provided = resolution_cache.provider_source(&id, target)?;
    let externally_loaded = provided.is_some();
    let source = match &provided {
        Some((source, _, _, _, _)) => source.clone(),
        None => fs::read_to_string(path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?,
    };
    frontend_profile::finish(Phase::Read, read_started);
    let hash = content_hash(source.as_bytes());
    // A `.vue`/`.svelte` component is compiled to JavaScript by the app's own
    // compiler before anything below reads it as a module (see
    // [`precompile_component`]).
    let (component_code, language, mut component, mut provider_assets) =
        if let Some((_, language, assets, watch_files, provider_diagnostics)) = provided {
            diagnostics.extend(provider_diagnostics);
            let mut side_effects = ComponentSideEffects::default();
            side_effects.css_source_files = watch_files;
            (None, language, side_effects, assets)
        } else {
            match diffpack_default_loader::module::precompile_component(
                path,
                &source,
                resolution_cache.css_preprocess.root_path(),
                resolution_cache.css_preprocess.postcss.as_deref(),
            )? {
                Some(compiled) => (
                    Some(compiled.code),
                    compiled.language,
                    compiled.side_effects,
                    Vec::new(),
                ),
                None => (
                    None,
                    diffpack_core::transform::SourceLanguage::FromPath,
                    ComponentSideEffects::default(),
                    Vec::new(),
                ),
            }
        };
    let module_text = component_code.as_deref().unwrap_or(source.as_str());
    let source = resolution_cache.apply_vite_replacements(path, module_text, target)?;
    let source_was_rewritten = matches!(source, Cow::Owned(_));
    let (source, language, provider_rewritten) = if externally_loaded {
        (source.into_owned(), language, false)
    } else {
        let before_provider = source.into_owned();
        let (source, language, assets, watch_files, provider_diagnostics) =
            resolution_cache.transform_external_source(&id, &before_provider, language, target)?;
        diagnostics.extend(provider_diagnostics);
        let rewritten = source != before_provider;
        provider_assets.extend(assets);
        component.css_source_files.extend(watch_files);
        (source, language, rewritten)
    };
    let project_config = diffpack_default_loader::jsx_project_config::project_config(
        resolver,
        path,
        &resolution_cache.jsx,
        compiler.is_generated_path(path),
    )?;
    let mut transformed = compiler.compile(CompileRequest {
        path,
        source: &source,
        target,
        hmr,
        refresh: hmr && target == Target::Client,
        jsx: resolution_cache.jsx_extensions,
        project_config: &project_config,
        language,
        source_maps,
    });
    // See the `&self` twin: a rewritten source is labelled as one.
    diffpack_default_loader::module::mark_rewritten_source(
        &mut transformed,
        component_code.is_some(),
        source_was_rewritten || provider_rewritten,
    );
    let code_hash = content_hash(transformed.code.as_bytes());
    diagnostics.extend(source_diagnostics(path, &transformed.diagnostics));
    let (dependency_specifiers, dependency_demands) = component.dependencies(&transformed);
    let dependencies = resolve_dependencies(
        resolver,
        resolution_cache,
        compiler,
        path,
        target,
        &dependency_specifiers,
        &dependency_demands,
        &mut diagnostics,
    );
    let droppable =
        diffpack_default_loader::side_effects::droppable_with_diagnostics(path, &mut diagnostics);
    Ok(LoadedModule {
        hash,
        code_hash,
        dependencies: dependencies.dependencies,
        pruned_imports: dependencies.pruned_imports,
        source: Arc::from(source),
        flat_module: transformed.flat_module,
        code: transformed.code,
        diagnostics,
        assets: component.assets,
        provider_assets,
        css: component.css,
        css_source_files: component.css_source_files,
        css_external_imports: component.css_external_imports,
        externals: dependencies.externals,
        droppable,
        liveness: transformed.liveness,
        uses_top_level_await: transformed.uses_top_level_await,
        uses_cjs_globals: transformed.uses_cjs_globals,
        uses_dirname: transformed.uses_dirname,
        workers: resolve_worker_entries(resolver, path, &transformed.workers)?,
        map: transformed.map,
    })
}

/// Loads a non-JavaScript module when a loader applies to `path`/`id`: a query
/// loader (`?url`, `?raw`), a global stylesheet (`.css`), or a default asset
/// import (image/font/SVG/...). Returns `None` for an ordinary JS/TS module,
/// which the normal read-and-transform path then handles.
fn load_special_module(
    id: &str,
    path: &Path,
    target: Target,
    hmr: bool,
    cache: &ResolutionCache,
    compiler: &dyn ModuleCompiler,
    policy: &dyn SpecialModulePolicy,
) -> Option<Result<SpecialModule, String>> {
    let resource = ResourceId::parse(id);
    let mut compile =
        |path: &Path, source: &str| compile_synthetic_with(compiler, path, source, target);
    if resource.query.is_some() {
        let result = (|| {
            if let Some(module) = diffpack_default_loader::module::query_module(
                &resource,
                &cache.base,
                cache.asset_inline_limit,
                cache.css_preprocess.root_path(),
                &mut compile,
            )? {
                return Ok(module);
            }
            policy
                .query_module(&resource, target, &mut compile)?
                .ok_or_else(|| loader_policy::unimplemented_error(&resource))
        })();
        return Some(result.map(|mut module| {
            policy.finalize_module(id, target, hmr, cache.jsx_extensions, &mut module);
            module
        }));
    }
    let postcss = cache.css_preprocess.postcss.as_deref();
    let result = diffpack_default_loader::module::path_module(
        path,
        postcss,
        &cache.scss,
        cache.css_preprocess.root_path(),
        &cache.base,
        cache.asset_inline_limit,
        &mut compile,
        |path, bytes, public_name| match cache.image_import_shape {
            ImageImportShape::Url => Ok(None),
            ImageImportShape::NextObject {
                responsive_variants,
            } => {
                let mut compile_image = |path: &Path, source: &str| {
                    compile_synthetic_with(compiler, path, source, Target::Server)
                };
                policy.asset_module(
                    path,
                    bytes,
                    public_name,
                    &cache.base,
                    responsive_variants,
                    &mut compile_image,
                )
            }
        },
    );
    result.map(|result| {
        result.map(|mut module| {
            policy.finalize_module(id, target, hmr, cache.jsx_extensions, &mut module);
            module
        })
    })
}

fn compile_synthetic_with(
    compiler: &dyn ModuleCompiler,
    path: &Path,
    source: &str,
    target: Target,
) -> diffpack_core::transform::TransformResult {
    compiler.compile(CompileRequest {
        path,
        source,
        target,
        hmr: false,
        refresh: false,
        jsx: diffpack_core::parser::JsxExtensions::default(),
        project_config: &diffpack_core::transform::ProjectConfig::default(),
        language: diffpack_core::transform::SourceLanguage::FromPath,
        source_maps: false,
    })
}

/// A Sass source (`.scss`): compiled natively to plain CSS first, then handed
/// to the SAME pipeline a hand-written CSS file takes — `.module.scss` through
/// the CSS Modules scoper, everything else through the global-stylesheet
/// loader. Every `@use`/`@import`ed partial (and the `additionalData` theme)
/// is recorded in `css_source_files`, so editing one re-derives this module.
/// A Less/Stylus source: compiled to plain CSS by the app's own preprocessor
/// (`node`, cwd = project root), then handed to the SAME pipeline a hand-written
/// CSS file takes — `.module.less`/`.module.styl` through the CSS Modules scoper,
/// everything else through the global-stylesheet loader. Every `@import`ed file
/// the preprocessor pulled in is recorded so editing it re-derives this module.
/// Builds the module for a query-bearing id. `?url` emits a content-hashed asset
/// and exports its URL; `?raw` inlines the file contents as a string.
/// Recognized-but-unimplemented loaders (`?tsr-split`) and unrecognized queries
/// produce a specific, actionable error rather than a misleading filesystem read
/// failure.
/// A content-hashed asset module: copies the file into `assets/` and exports its
/// public URL as the default export. Used for both `?url` and default asset
/// imports (images, fonts, SVG, ...).
///
/// Under [`ImageImportShape::NextObject`] a decodable PNG/JPEG default import
/// materializes as Next's static-image object (`{ src, width, height,
/// blurDataURL, variants }`) with build-emitted responsive variants, instead of a
/// bare URL string. Every other shape/format keeps the bare-URL behavior
/// byte-for-byte, so Vite/TanStack/generic builds are unaffected.
/// The directory a Tailwind entry's candidate scan covers. Tailwind v4's
/// default source detection scans the PROJECT, not the stylesheet's own
/// directory: an entry that declares `source(...)` keeps its explicit root;
/// otherwise walk up to the nearest `package.json` (found live: wall-go keeps
/// its entry in `src/assets/`, and scanning only that directory yielded zero
/// utility candidates — an entirely unstyled app).
/// A content hash of anything hashable, for cache keys whose inputs are too large to
/// keep but must be compared exactly.
fn hash_of(value: &impl std::hash::Hash) -> u64 {
    use std::hash::Hasher;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

use diffpack_default_loader::asset::content_hash;

fn resolve_dependencies(
    resolvers: &Resolvers,
    resolution_cache: &ResolutionCache,
    compiler: &dyn ModuleCompiler,
    path: &Path,
    target: Target,
    dependency_specifiers: &[String],
    dependency_demands: &[DependencyDemand],
    diagnostics: &mut Vec<Diagnostic>,
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
        let recorded_demand = dependency_demands
            .iter()
            .find(|demand| demand.specifier == *specifier);
        let demand_of = || {
            recorded_demand
                .cloned()
                .unwrap_or_else(|| DependencyDemand {
                    specifier: specifier.clone(),
                    all: true,
                    names: Vec::new(),
                    dynamic: false,
                    optional: false,
                    // Nothing recorded a syntax for this specifier, so it is not a
                    // `require(...)` call site: it came from a loader or an injected
                    // import, both of which are ESM.
                    require_syntax: false,
                    import_syntax: true,
                    // A loader/injected import is a STATIC import: it is not an
                    // `import()` call site, so the target can never be deferred.
                    eager: true,
                })
        };
        // Which export conditions answer this specifier. A `require(...)` and an
        // `import` of the SAME specifier from the SAME file are two different
        // resolutions, so a module that does both is resolved both ways and the
        // two answers must agree — see [`same_specifier_resolves_two_ways_message`].
        let syntax = match recorded_demand {
            Some(demand) if demand.require_syntax && !demand.import_syntax => {
                ImportSyntax::CommonJs
            }
            _ => ImportSyntax::Esm,
        };
        let also_resolve_as_common_js =
            recorded_demand.is_some_and(|demand| demand.require_syntax && demand.import_syntax);
        // An external (a Node built-in like `node:stream`) is not a graph module:
        // it is neither resolved nor bundled, and its `require(...)` is left in
        // place for the runtime to resolve. On a SERVER target that is correct and
        // not a diagnostic — Node resolves it. On a BROWSER target there is no
        // runtime that can: the emitted `require` hits the throw-on-use stub and
        // the page dies, while the build exits 0. That is the same
        // silent-broken-artifact class as an unresolved import, so it is fatal.
        // `serverExternalPackages`: the build was told this package stays a runtime
        // `require` from `node_modules`. It is therefore not a graph module at all —
        // never resolved, never bundled, never diagnosed. A CLIENT build ignores the
        // list: a browser has no `node_modules` to require from, so externalizing there
        // would emit a chunk that dies on the throw-on-use stub.
        if target != Target::Client && resolution_cache.is_external_package(specifier) {
            if !externals.iter().any(|existing| existing == specifier) {
                externals.push(specifier.clone());
            }
            continue;
        }
        if diffpack_default_loader::resolver_policy::is_external_specifier(specifier) {
            // An alias may deliberately map a built-in onto a browser polyfill —
            // Next does exactly this for `url`, `querystring`, `buffer`, ... using the
            // copies it vendors, so such an import is valid in a Next client page. The
            // resolver carries the project's alias table, so a built-in specifier that
            // RESOLVES to a real file was mapped on purpose: bundle that file instead of
            // rejecting it. A built-in with no mapping still falls through below.
            if target == Target::Client
                && let Ok(resolved) = directory_cache.resolve(resolvers, path, specifier, syntax)
            {
                dependencies.push((specifier.clone(), resolved.id, demand_of()));
                continue;
            }
            if target == Target::Client {
                diagnostics.push(Diagnostic {
                    kind: DiagnosticKind::NodeBuiltinInBrowser {
                        specifier: specifier.clone(),
                        importer: path.to_path_buf(),
                    },
                    message: node_builtin_in_browser_message(path, specifier),
                });
                continue;
            }
            if !externals.iter().any(|existing| existing == specifier) {
                externals.push(specifier.clone());
            }
            continue;
        }
        match directory_cache.resolve(resolvers, path, specifier, syntax) {
            Ok(resolved) => {
                if resolved.provider_external {
                    let external = resolved.id.to_string();
                    if !externals.contains(&external) {
                        externals.push(external);
                    }
                    continue;
                }
                let demand = demand_of();
                // The module reaches this specifier BOTH ways. Node treats that as
                // two modules whenever the package's `exports` map sends `import`
                // and `require` to different files, and the emitted runtime map has
                // one entry per specifier — one target — so it cannot express two.
                // Silently picking either would give some call site the wrong
                // module, which is precisely the failure this resolution split
                // exists to end, so it is fatal and names both files.
                if also_resolve_as_common_js
                    && let Ok(as_common_js) =
                        directory_cache.resolve(resolvers, path, specifier, ImportSyntax::CommonJs)
                    && as_common_js.id != resolved.id
                {
                    diagnostics.push(Diagnostic {
                        kind: DiagnosticKind::SpecifierResolvesTwoWays {
                            specifier: specifier.clone(),
                            importer: path.to_path_buf(),
                        },
                        message: specifier_resolves_two_ways_message(
                            path,
                            specifier,
                            Path::new(resolved.id.as_ref()),
                            Path::new(as_common_js.id.as_ref()),
                        ),
                    });
                }
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
            // A dependency the module itself declares recoverable — every reference
            // to it is a `require(...)` inside a `try` — is reported but not fatal.
            // Node would throw MODULE_NOT_FOUND at that exact `require`, and the
            // emitted bundle does the same (`requireNative` throws immediately for a
            // specifier with no map entry), so the module's `catch` runs and the
            // artifact behaves as its author wrote it. See
            // [`DiagnosticKind::OptionalDependencyMissing`].
            Err(_)
                if dependency_demands
                    .iter()
                    .any(|demand| demand.specifier == *specifier && demand.optional) =>
            {
                diagnostics.push(Diagnostic {
                    kind: DiagnosticKind::OptionalDependencyMissing {
                        specifier: specifier.clone(),
                        importer: path.to_path_buf(),
                    },
                    message: optional_dependency_missing_message(path, specifier),
                });
            }
            // A scheme-qualified specifier naming ANOTHER runtime's built-in module.
            // `node:fs` is handled above as an external because Node provides it;
            // `cloudflare:sockets`, `bun:ffi` and friends are the identical shape for
            // a different host, and diffpack's rule was accidentally Node-only. No
            // filesystem lookup can satisfy one and no registry can install it, so
            // `npm install cloudflare:sockets` was not merely unhelpful, it was wrong.
            // Treat it exactly like a built-in: external on a SERVER graph (the host
            // resolves it if it is that host; otherwise the import throws right where
            // it would under plain Node), fatal on a browser graph — which is where
            // the arm below still catches it, because a browser has no such host.
            Err(_) if target != Target::Client && host_provided_scheme(specifier).is_some() => {
                diagnostics.push(Diagnostic {
                    kind: DiagnosticKind::HostProvidedModule {
                        specifier: specifier.clone(),
                        importer: path.to_path_buf(),
                    },
                    message: host_provided_module_message(path, specifier),
                });
                if !externals.iter().any(|existing| existing == specifier) {
                    externals.push(specifier.clone());
                }
            }
            Err(error) => diagnostics.push(Diagnostic {
                kind: DiagnosticKind::UnresolvedImport {
                    specifier: specifier.clone(),
                    importer: path.to_path_buf(),
                },
                message: diffpack_default_loader::resolution_diagnostic::unresolved_import_message(
                    path,
                    specifier,
                    &error.to_string(),
                    compiler.is_generated_path(path),
                    compiler.unresolved_import_help(specifier),
                ),
            }),
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

/// Resolves a synthesized module's carried import specifiers into real graph
/// edges. A leaf synthetic module (asset URL, `?raw`, stylesheet) carries none
/// and resolves to nothing. A route-split (`?tsr-split`) module carries the
/// imports of the extracted route property, which must be resolved relative to
/// the REAL source file (the route file), not the virtual `id` that still bears
/// the `?tsr-split=…` query — so the split module links to the same React and
/// route-level modules every other importer sees, and its lowered `require(...)`
/// calls get real runtime map entries instead of falling through to
/// `requireNative`.
fn resolve_special_dependencies(
    resolvers: &Resolvers,
    resolution_cache: &ResolutionCache,
    compiler: &dyn ModuleCompiler,
    id: &str,
    target: Target,
    special: &SpecialModule,
    diagnostics: &mut Vec<Diagnostic>,
) -> ResolvedDependencies {
    if special.dependency_specifiers.is_empty() {
        return ResolvedDependencies {
            dependencies: Vec::new(),
            pruned_imports: HashSet::new(),
            externals: Vec::new(),
        };
    }
    // A build-generated virtual module's id is a synthetic specifier, not a real
    // path, so it has no directory to resolve bare-package imports from. Resolve its
    // dependencies as if the module lived in the project (the entry's directory), so
    // `react-server-dom-webpack/client` and the like resolve against `node_modules`.
    let source_file = if resolution_cache.virtual_modules.contains_key(id) {
        resolution_cache
            .virtual_import_base
            .join("__diffpack_virtual_module__.js")
    } else {
        PathBuf::from(ResourceId::parse(id).path)
    };
    resolve_dependencies(
        resolvers,
        resolution_cache,
        compiler,
        &source_file,
        target,
        &special.dependency_specifiers,
        &special.dependency_demands,
        diagnostics,
    )
}

/// A per-chunk render cache, keyed by a stable `Bundler::chunk_render_key`: the
/// chunk's ordered dense-module ids, each member's transformed-content hash, and
/// every render input that affects the emitted bytes (format, `is_entry`, and the
/// `chunk_names`/`runtime_ids`/export-demand entries the chunk references). A hit
/// is byte-identical to a fresh `render_best`, so a leaf edit re-renders only the
/// one chunk whose key changed; every other chunk is reused verbatim.
///
/// The cache is bounded to the currently-live chunk set: each emit records the
/// keys it used and evicts every entry not among them, so retained bytes stay
/// flat across a long edit sequence (a chunk that stops being reachable, or whose
/// content changes, drops its old entry). This upholds the memory guards in
/// `docs/THESIS_GUARDS.md`.
/// What a single [`Bundler::emit_with_options`] wrote and re-rendered. The
/// `rendered_chunks` count is the incrementality signal (a leaf edit re-renders
/// exactly one chunk); `written` is the set of files kept on disk, so the
/// environment emit can delete only files that are no longer part of the build
/// instead of nuking the whole output tree.
#[derive(Debug, Default)]
pub struct EmitStats {
    pub rendered_chunks: usize,
    #[doc(hidden)]
    pub written: BTreeSet<PathBuf>,
    /// Set when the emit stopped early because its [`EmitCancel`] fired. Whatever
    /// was written is complete and valid (chunks are written whole); what was not
    /// written is simply not there yet, so the caller must neither prune against
    /// this emit's written set nor consider the graph's debt discharged.
    pub cancelled: bool,
}

/// What a stylesheet-only emit did.
pub enum StylesheetEmit {
    /// The sheet was compiled; this is the file it was written to (unchanged bytes are
    /// not rewritten, so the path may be untouched on disk).
    Written(PathBuf),
    /// This graph compiles no CSS at all.
    NoStylesheet,
    /// Abandoned before finishing — see [`EmitCancel`]. Nothing was written, and the
    /// caller must keep the work owed.
    Cancelled,
}

/// A "stop as soon as you can" signal for an emit.
///
/// The dev loop's chunk compaction is housekeeping that runs on the same thread that
/// answers file events, and on a large app one chunk render is hundreds of
/// milliseconds. Without a way to abandon it, an edit that lands mid-compaction waits
/// it out — the contention cliff that a long fixed idle only made rarer, never
/// impossible. With this, the loop asks "has a file event arrived?" between renders
/// and inside the per-module render fan-out, and drops the pass within a millisecond
/// or two, keeping its debt for the next quiet moment.
///
/// Every other caller passes [`EmitCancel::never`], which compiles to one `Option`
/// test per check.
/// Whether `name` is a JavaScript identifier. A source map's `names` entry means
/// "the identifier this position had in the original source"; Oxc records the raw
/// source text under the printed node's span, which is that identifier for an
/// identifier node and arbitrary source text for anything else (a whole
/// `import("./x")` expression, say). Publishing the latter would put text in
/// `names` that is not a name — and, for a source that embeds one, an absolute
/// path. So only real identifiers are published.
/// What one minified-map token resolved to during source-map composition: the
/// module whose readable token it landed on, that token's original position, and
/// the name (already chosen and verified) the composed map should carry there.

/// The longest shared leading directory of two absolute paths. Used to derive a
/// project root for project-relative source-map labels.
fn common_ancestor(left: &Path, right: &Path) -> PathBuf {
    let mut result = PathBuf::new();
    for (left_component, right_component) in left.components().zip(right.components()) {
        if left_component != right_component {
            break;
        }
        result.push(left_component.as_os_str());
    }
    result
}

fn path_with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_owned();
    value.push(suffix);
    PathBuf::from(value)
}

/// How a default asset import of a raster image (`import img from './x.png'`)
/// materializes. `Url` (the default, and what Vite/TanStack/generic builds use)
/// makes the default export the bare public URL string, byte-identical to Vite.
/// `NextObject` makes it Next's static-import object shape
/// (`{ src, width, height, blurDataURL, variants }`) with build-emitted responsive
/// variants — set ONLY by the Next app-router adapter so no other build path
/// changes its asset-import semantics.
pub use diffpack_default_loader::{CssPreprocess, ImageImportShape};

/// The two ESM-only export conditions. A `require(...)` call site must not
/// resolve under either: `package.json`'s `exports` is a MAP from condition to
/// file, so leaving `import` in the set for a `require` resolution picks the ESM
/// file whenever the package lists `import` first — which is the whole reason
/// dual-package publishing works at all.
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
