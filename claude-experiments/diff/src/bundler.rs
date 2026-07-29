use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fs;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use oxc_resolver::{ResolveError, ResolveOptions, Resolver, SideEffects, TsconfigDiscovery};
use oxc_sourcemap::SourceMapBuilder;

use crate::source_map::{LineTrack, MapOrigin, MapToken, ModuleSourceMap};
#[allow(unused_imports)]
use crate::source_map::ColumnEdit;
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::frontend_profile::{self, Phase};
use crate::resource_id::{LoaderKind, ResourceId};
use crate::transform::{
    DependencyDemand, FlatModule, FoldExpression, ModuleLiveness, Target, transform_module,
};

pub type ModuleId = String;
type DenseModuleId = usize;
type SharedModuleId = Arc<str>;

#[derive(Debug, Clone)]
struct ModuleState {
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
    uses_import_meta: bool,
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
    css: Option<String>,
    css_source_files: Vec<PathBuf>,
    css_external_imports: Vec<String>,
    externals: Vec<String>,
    droppable: bool,
    liveness: ModuleLiveness,
    uses_top_level_await: bool,
    uses_import_meta: bool,
    uses_cjs_globals: bool,
    uses_dirname: bool,
    /// The module's REAL source map over `code`; see [`ModuleState::map`].
    map: Option<ModuleSourceMap>,
    /// Module-worker entries this module creates: `(placeholder_key,
    /// resolved_entry_path)`. Emitted as self-contained bundles under
    /// `assets/`; the key ties the code placeholder to the emitted file.
    workers: Vec<(String, PathBuf)>,
}

/// A static asset (e.g. a `?url` import target) that must be content-hashed and
/// copied into the output `assets/` directory. The synthetic JavaScript module
/// that references it exports the public URL `/assets/<public_name>`.
#[derive(Debug, Clone)]
struct AssetEmit {
    source: PathBuf,
    public_name: String,
    /// A Tailwind v4 CSS entry (`@import 'tailwindcss'`) imported for its URL.
    /// Rather than copying the raw source (which would leave the browser fetching
    /// `@import 'tailwindcss'` and 404ing), the emit step compiles it natively
    /// against the class candidates scanned from the reachable source graph. The
    /// stored string is the raw CSS source (captured at load); `None` for an
    /// ordinary asset that is copied verbatim.
    tailwind_source: Option<String>,
    /// Responsive downscale widths to emit alongside the copied original, for a
    /// Next static-image import (`import img from './x.png'` under
    /// [`ImageImportShape::NextObject`]). Each width `w` is written as
    /// `<stem>-<w>.<ext>` next to the content-hashed original — the SAME native
    /// resize the public-image path uses, run once at emit time (off the transform
    /// hot path, mirroring the `tailwind_source` special-case). `None` for an
    /// ordinary asset (no variants).
    image_variants: Option<Vec<u32>>,
}

struct ResolutionCache {
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
    import_meta_env: Arc<Option<crate::import_meta_env::ImportMetaEnv>>,
    /// Vite `import.meta.glob` expansion, when opted in. Applied alongside
    /// `import_meta_env` on both load paths; `None` leaves `import.meta.glob`
    /// untouched (generic bundling). See [`crate::import_meta_glob`].
    import_meta_glob: Arc<Option<crate::import_meta_glob::ImportMetaGlob>>,
    /// Vite `define` replacements, when opted in. Applied alongside
    /// `import_meta_env` on both load paths.
    defines: Arc<Vec<(String, String)>>,
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
    scss: Arc<crate::sass::ScssOptions>,
    /// How default raster-image imports materialize (Vite bare-URL string vs the
    /// Next static-import object shape). Threaded to `synthesize_asset_url`.
    image_import_shape: ImageImportShape,
    /// Less/Stylus + PostCSS wiring, threaded to the CSS loaders.
    css_preprocess: CssPreprocess,
    /// The project's JSX-extension rule, threaded to the module transform on both
    /// the serial ([`Bundler::load_module`]) and parallel ([`load_uncached`]) load
    /// paths — the one parse whose diagnostics the build reports.
    jsx_extensions: crate::parser::JsxExtensions,
    /// The BUILD's JSX lowering settings (`vite.config`'s `esbuild.*` / `oxc.jsx`).
    /// Layered over each file's owning tsconfig by [`jsx_config_for`] on both load
    /// paths; empty (the default) leaves the tsconfig — and, failing that, oxc's
    /// react-automatic default — in charge.
    jsx: Arc<crate::transform::JsxConfig>,
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
struct BrowserFieldMap {
    /// Whether this build honours the field at all. Only a browser target does;
    /// a server build wants the real Node entries, so the fast path is always
    /// free there.
    honored: bool,
    directories: Mutex<HashMap<PathBuf, bool>>,
}

impl BrowserFieldMap {
    fn new(honored: bool) -> Self {
        Self {
            honored,
            directories: Mutex::new(HashMap::new()),
        }
    }

    /// Whether a file directly in `directory` may be rewritten by an enclosing
    /// package's object-form `browser` field.
    fn remaps_directory(&self, directory: &Path) -> bool {
        if !self.honored {
            return false;
        }
        if let Some(known) = self
            .directories
            .lock()
            .expect("browser-field directory cache poisoned")
            .get(directory)
        {
            return *known;
        }
        let answer = nearest_package_has_object_browser_field(directory);
        self.directories
            .lock()
            .expect("browser-field directory cache poisoned")
            .insert(directory.to_path_buf(), answer);
        answer
    }
}

/// Whether the nearest `package.json` at or above `directory` — the one that
/// DESCRIBES these files, exactly as Node's resolver picks it — carries a
/// `browser` field in its object form.
///
/// A manifest that exists but cannot be read or parsed answers `true`: that
/// merely routes the specifier through the real resolver, which reads the same
/// file and reports the problem properly. Guessing `false` would silently pick
/// the unmapped file instead.
fn nearest_package_has_object_browser_field(directory: &Path) -> bool {
    for ancestor in directory.ancestors() {
        let manifest = ancestor.join("package.json");
        if !manifest.is_file() {
            continue;
        }
        let Ok(text) = fs::read_to_string(&manifest) else {
            return true;
        };
        // The nearest manifest decides; the walk stops here either way.
        if !text.contains("\"browser\"") {
            return false;
        }
        let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&text) else {
            return true;
        };
        return manifest.get("browser").is_some_and(serde_json::Value::is_object);
    }
    false
}

/// Which syntax reaches a specifier, and so which export conditions resolve it.
///
/// `package.json`'s `exports` is a map keyed by condition, and `import` and
/// `require` are different keys pointing at different files in almost every
/// dual-published package. Resolving both the same way is not an approximation,
/// it hands back the wrong module: `pg/lib/index.js` does `require('pg-pool')`
/// and then `class Pool extends …`, and pg-pool's ESM entry is a Module
/// namespace object, which is not a constructor.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum ImportSyntax {
    /// A static `import` / `export … from`, or a dynamic `import()`.
    Esm,
    /// A CommonJS `require(...)` call.
    CommonJs,
}

/// The resolvers a build needs: one per [`ImportSyntax`]. They differ ONLY in
/// their export conditions (see [`resolve_options_for_syntax`]), so they share
/// every other resolution rule and each keeps its own oxc_resolver cache.
///
/// Derefs to the ESM resolver, which is the right answer for every caller that
/// resolves something the build itself synthesized (a worker entry, a tsconfig
/// lookup) rather than a specifier written in a module.
pub(crate) struct Resolvers {
    esm: Resolver,
    common_js: Resolver,
}

impl Resolvers {
    fn new(config: &BuildConfig) -> Self {
        Self {
            esm: Resolver::new(resolve_options_for_syntax(config, ImportSyntax::Esm)),
            common_js: Resolver::new(resolve_options_for_syntax(config, ImportSyntax::CommonJs)),
        }
    }

    fn for_syntax(&self, syntax: ImportSyntax) -> &Resolver {
        match syntax {
            ImportSyntax::Esm => &self.esm,
            ImportSyntax::CommonJs => &self.common_js,
        }
    }
}

impl std::ops::Deref for Resolvers {
    type Target = Resolver;

    fn deref(&self) -> &Resolver {
        &self.esm
    }
}

struct DirectoryResolutionCache {
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
}

impl ResolutionCache {
    #[allow(clippy::too_many_arguments)]
    fn new(
        aliases: Vec<(String, PathBuf)>,
        virtual_modules: Vec<(String, String)>,
        import_meta_env: Option<crate::import_meta_env::ImportMetaEnv>,
        import_meta_glob: Option<crate::import_meta_glob::ImportMetaGlob>,
        defines: Vec<(String, String)>,
        base: &str,
        virtual_import_base: PathBuf,
        asset_inline_limit: usize,
        scss: crate::sass::ScssOptions,
        image_import_shape: ImageImportShape,
        css_preprocess: CssPreprocess,
        jsx_extensions: crate::parser::JsxExtensions,
        jsx: crate::transform::JsxConfig,
        honors_browser_field: bool,
        external_packages: Vec<String>,
    ) -> Self {
        Self {
            external_packages: Arc::new(external_packages),
            browser_field: Arc::new(BrowserFieldMap::new(honors_browser_field)),
            directories: std::array::from_fn(|_| Mutex::new(HashMap::new())),
            aliases: Arc::new(aliases),
            virtual_modules: Arc::new(virtual_modules.into_iter().collect()),
            import_meta_env: Arc::new(import_meta_env),
            import_meta_glob: Arc::new(import_meta_glob),
            defines: Arc::new(defines),
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
        if let Some(source) = crate::runtime_helpers::helper_source(id) {
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
        if let Some(rewritten) = crate::dynamic_import_context::transform(path, &current) {
            current = Cow::Owned(rewritten);
        }
        // Glob expansion first: it emits imports the rest of the pipeline (and the
        // format-sensitive `import.meta` scan) must see as ordinary graph edges.
        // A malformed call is a hard build error, never a silently empty object.
        if let Some(options) = self.import_meta_glob.as_ref()
            && let Some(rewritten) = crate::import_meta_glob::transform(path, &current, options)?
        {
            current = Cow::Owned(rewritten);
        }
        if let Some(options) = self.import_meta_env.as_ref()
            && let Some(rewritten) = crate::import_meta_env::transform(path, &current, options, target)
        {
            current = Cow::Owned(rewritten);
        }
        if !self.defines.is_empty()
            && let Some(rewritten) = crate::vite_define::transform(path, &current, &self.defines)
        {
            current = Cow::Owned(rewritten);
        }
        // Only worth attempting when a substitution above could have made a test
        // decidable; a module nobody rewrote keeps its branches untouched.
        if !matches!(current, Cow::Borrowed(_))
            && std::env::var_os("DIFFPACK_DISABLE_DEAD_BRANCH").is_none()
            && let Some(rewritten) = crate::dead_branch::transform(path, &current)
        {
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
        if crate::runtime_helpers::helper_name(specifier).is_some() {
            let result = if crate::runtime_helpers::helper_source(specifier).is_some() {
                Ok(ResolvedModule {
                    id: SharedModuleId::from(specifier),
                    side_effect_free: true,
                })
            } else {
                Err(crate::runtime_helpers::unknown_helper_error(specifier))
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
            });
            shard
                .lock()
                .expect("resolution specifier cache poisoned")
                .insert(specifier.to_owned(), result.clone());
            return result;
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
        let relative = path_specifier.strip_prefix('/').filter(|rest| !rest.is_empty())?;
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

/// Why a build diagnostic was produced, and therefore whether the artifact it
/// describes is broken or merely imperfect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiagnosticKind {
    /// An import specifier that resolved to nothing. The specifier gets no
    /// runtime-map entry, so the emitted chunk carries a dangling `require`
    /// that throws the moment the module loads. The artifact is broken.
    UnresolvedImport {
        specifier: String,
        importer: PathBuf,
    },
    /// A Node built-in reached from a BROWSER graph. Leaving it external emits a
    /// `require` that no browser can satisfy, so the page dies at runtime while
    /// the build exits 0. Same class as [`DiagnosticKind::UnresolvedImport`]:
    /// the artifact is broken.
    NodeBuiltinInBrowser {
        specifier: String,
        importer: PathBuf,
    },
    /// An oxc parse/semantic/transform diagnostic. At error severity the emitted
    /// code does not match the source; a warning leaves runnable code.
    Source { fatal: bool },
    /// A `package.json` `sideEffects` glob this matcher cannot evaluate. The
    /// module is KEPT (see [`module_droppable`]), so the bundle is correct, just
    /// larger. `"sideEffects": ["*.{css,scss}"]` is a common idiom; failing on it
    /// would reject apps that bundle perfectly well.
    SideEffectsGlob,
    /// An OPTIONAL dependency that resolved to nothing: every reference to it is a
    /// `require(...)` inside a `try` block (see
    /// [`crate::parser::collect_optional_dependencies`]). Unlike
    /// [`DiagnosticKind::UnresolvedImport`] the artifact is NOT broken — the emitted
    /// `require` throws exactly where Node's would, and the module's own `catch`
    /// takes over, which is the behaviour the package was written for. Reported so
    /// the omission is visible, never fatal.
    OptionalDependencyMissing {
        specifier: String,
        importer: PathBuf,
    },
    /// A `scheme:`-qualified specifier naming another runtime's built-in module
    /// (`cloudflare:sockets`, `bun:ffi`), left external on a SERVER graph exactly as
    /// `node:fs` is. The artifact is not broken: on that host the import resolves, and
    /// anywhere else it throws at the same point it would without a bundler. Reported
    /// so an external is never silent.
    HostProvidedModule {
        specifier: String,
        importer: PathBuf,
    },
    /// One module reaches one specifier through BOTH a `require(...)` and an ESM
    /// `import`, and the package's `exports` map sends the two syntaxes to
    /// different files. Node would load two distinct modules here; the emitted
    /// runtime map holds one target per specifier, so whichever is chosen is
    /// wrong for the other call site. The artifact is broken either way, so this
    /// is fatal.
    SpecifierResolvesTwoWays {
        specifier: String,
        importer: PathBuf,
    },
}

/// One build diagnostic. Typed rather than a bare string so fatality is a
/// property of the diagnostic, not a substring match at each consumer.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub kind: DiagnosticKind,
    pub message: String,
}

impl Diagnostic {
    /// The single fatality predicate: a diagnostic is fatal iff the emitted
    /// artifact would be WRONG, not merely bigger or noisier.
    pub fn is_fatal(&self) -> bool {
        match &self.kind {
            DiagnosticKind::UnresolvedImport { .. } => true,
            DiagnosticKind::NodeBuiltinInBrowser { .. } => true,
            DiagnosticKind::Source { fatal } => *fatal,
            DiagnosticKind::SideEffectsGlob => false,
            DiagnosticKind::OptionalDependencyMissing { .. } => false,
            DiagnosticKind::HostProvidedModule { .. } => false,
            DiagnosticKind::SpecifierResolvesTwoWays { .. } => true,
        }
    }
}

/// Splits a build's diagnostics into the non-fatal warnings (`Ok`) and, when any
/// diagnostic is fatal, one error message naming EVERY fatal diagnostic (`Err`)
/// so a build reports all of them at once instead of only the first. `context`
/// names what was being built, e.g. `"react-server build"`.
pub fn partition_diagnostics(
    diagnostics: &[Diagnostic],
    context: &str,
) -> Result<Vec<String>, String> {
    let (fatal, warnings): (Vec<_>, Vec<_>) =
        diagnostics.iter().partition(|diagnostic| diagnostic.is_fatal());
    if fatal.is_empty() {
        return Ok(warnings
            .into_iter()
            .map(|diagnostic| diagnostic.message.clone())
            .collect());
    }
    // The consequence sentence has to match the fatalities actually present. A
    // dangling reference is what an UNRESOLVED IMPORT would leave behind; a
    // source error means the module never compiled at all, so claiming a dangling
    // reference sends the reader hunting for an import that is perfectly fine.
    let dangling = fatal.iter().any(|diagnostic| {
        matches!(
            diagnostic.kind,
            DiagnosticKind::UnresolvedImport { .. } | DiagnosticKind::NodeBuiltinInBrowser { .. }
        )
    });
    let unparsed = fatal
        .iter()
        .any(|diagnostic| matches!(diagnostic.kind, DiagnosticKind::Source { .. }));
    let consequence = match (dangling, unparsed) {
        (true, true) => {
            "An artifact missing code diffpack could not compile, with dangling references to \
             the rest, would crash at runtime"
        }
        (true, false) => "An artifact with dangling references would crash at runtime",
        (false, _) => "The emitted code would not match the source",
    };
    let mut message = format!(
        "{context}: {} fatal build diagnostic(s). {consequence}, so no output was written.",
        fatal.len()
    );
    for diagnostic in fatal {
        message.push_str("\n\n  ");
        message.push_str(&diagnostic.message.replace('\n', "\n  "));
    }
    Err(message)
}

#[derive(Debug)]
pub struct BuildUpdate {
    pub delta: GraphDelta,
    pub transformed_modules: usize,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Debug, Clone, Default)]
pub struct GraphDelta {
    pub edge_updates: Vec<((ModuleId, ModuleId), isize)>,
    pub changed: BTreeSet<ModuleId>,
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ModuleFormat {
    /// CommonJS-shaped output: `module.exports`, `require`, `require.dynamic`.
    #[default]
    Cjs,
    /// Node ES module output: `export default`, real `import()` for split
    /// chunks, `createRequire(import.meta.url)` for external Node built-ins.
    Esm,
    /// Browser ES module output: like [`ModuleFormat::Esm`] but without the
    /// `node:module`/`createRequire` import, so the module loads in a browser.
    /// Node built-in externals resolve to a throw-on-call stub.
    BrowserEsm,
}

impl ModuleFormat {
    /// Whether this format emits ES module syntax (`export default`, native
    /// dynamic `import()` of split chunks). Both the Node and browser ESM
    /// variants share the same module boundary and dynamic-import lowering; they
    /// differ only in how the main chunk binds `requireNative`.
    fn is_esm(self) -> bool {
        matches!(self, ModuleFormat::Esm | ModuleFormat::BrowserEsm)
    }
}

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
const BROWSER_GLOBALS_PRELUDE: &str = "globalThis.process=globalThis.process||{};globalThis.process.env=globalThis.process.env||{};globalThis.process.env.NODE_ENV=globalThis.process.env.NODE_ENV||\"production\";globalThis.process.env.TSS_SERVER_FN_BASE=globalThis.process.env.TSS_SERVER_FN_BASE||\"/_serverFn/\";\n";

#[derive(Debug, Clone, Copy, Default)]
pub struct EmitOptions {
    pub source_map: bool,
    pub minify: bool,
    /// The target module system. Defaults to [`ModuleFormat::Cjs`]; the server
    /// build forces [`ModuleFormat::Esm`] so its `.mjs` output truly executes.
    pub format: ModuleFormat,
    /// Emit the dev-only Hot Module Replacement runtime (accept/dispose tracking,
    /// per-module `module.hot`, factory `replace`, and cache invalidation with
    /// chunk cache-busting). ALWAYS `false` for `build-app` production output, so
    /// production bundles are byte-for-byte unaffected; the dev server sets it.
    pub hmr: bool,
}

/// A count of what an environment build wrote to disk: JavaScript modules
/// (`.js` for the browser `public/` build, `.mjs` for the Node ESM `server/`
/// build), extracted stylesheets, and content-hashed assets. Counted from the
/// emitted files, not predicted, so the summary always matches reality.
#[derive(Debug, Clone)]
pub struct EmitSummary {
    pub output_dir: PathBuf,
    pub javascript_files: usize,
    pub css_files: usize,
    pub asset_files: usize,
    /// How many chunks this emit actually re-rendered (vs. reused byte-for-byte
    /// from the per-chunk render cache). Zero for a from-scratch `EmitSummary::of`
    /// walk; set by [`Bundler::emit_public`]/[`Bundler::emit_server`] from the
    /// underlying [`EmitStats`]. This is the incrementality signal a long-lived
    /// dev server reports per edit (a leaf edit re-renders exactly one chunk).
    pub rendered_chunks: usize,
}

impl EmitSummary {
    /// Walks an emitted environment directory and classifies each file: anything
    /// under `assets/` is a content-hashed asset; otherwise a `.js`/`.mjs`
    /// module or a `.css` stylesheet by extension. Files with any other
    /// extension are ignored (there are none today, but the count stays honest
    /// if that changes).
    fn of(output_dir: &Path) -> Result<Self, String> {
        let mut summary = Self {
            output_dir: output_dir.to_path_buf(),
            javascript_files: 0,
            css_files: 0,
            asset_files: 0,
            rendered_chunks: 0,
        };
        let mut stack = vec![output_dir.to_path_buf()];
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
                        Some("js" | "mjs") => summary.javascript_files += 1,
                        Some("css") => summary.css_files += 1,
                        _ => {}
                    }
                }
            }
        }
        Ok(summary)
    }
}

/// DEV-ONLY: where a changed module lives in the current emit, for pushing a
/// targeted HMR update. See [`Bundler::hmr_locate`].
#[derive(Debug, Clone)]
pub struct HmrLocation {
    /// The changed module's canonical id (path / virtual id).
    pub module_id: String,
    /// Its stable dense runtime id in the emitted registry.
    pub runtime_id: usize,
    /// The chunk file whose re-import re-registers this module's factory
    /// (`client.js` for entry-bundled code, `client.chunk-<n>.js` for a split).
    pub chunk_file: String,
}

/// One emitted non-main chunk in the partition [`Bundler::chunk_plan`] computes.
///
/// The plan's chunks are DISJOINT and, together with the main chunk, cover every
/// live module exactly once. That is what makes the output a partition rather than
/// a pile of overlapping closures: before this, each dynamic root emitted its
/// entire static closure, so anything two routes shared (React, the router core)
/// was duplicated into every chunk that reached it.
#[derive(Debug, Clone)]
struct ChunkPlan {
    /// Members in the graph's canonical (id-sorted) order.
    members: Vec<DenseModuleId>,
    /// The dynamic-import roots whose own module landed in this chunk. A purely
    /// shared chunk has none: no `import()` names it, it is only ever pulled in as
    /// another chunk's prerequisite.
    roots: Vec<DenseModuleId>,
    /// Indices, into the plan vector, of the chunks that must be evaluated before
    /// this one because a member statically depends on one of their members. The
    /// main chunk is never listed: it always loads first (it builds the runtime).
    prerequisites: Vec<usize>,
    /// The chunk's emitted file name (`client.chunk-3.js`, `client.shared-1.js`).
    file_name: String,
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

/// The whole-graph derivation every emit performs before it renders a single chunk:
/// which modules survive export-level dead-module elimination, the dense order that
/// fixes every runtime id, and the chunk partition each module lands in.
///
/// Deriving it walks the entire graph — on cal.com's 18 MB client graph,
/// `live_modules` is ~12 ms and `chunk_plan` ~25 ms — and every byte a later HMR
/// micro-chunk emits has to agree with it. Both facts point the same way: derive it
/// once, in the emit that produced the bundle the browser and the dev server's Node
/// processes are actually running, and reuse it verbatim. See [`Bundler::emit_plan`].
struct EmitPlan {
    /// Live modules in emit order. A module's index here IS its runtime id.
    reachable_dense: Vec<DenseModuleId>,
    /// The same set, for membership tests.
    allowed: HashSet<DenseModuleId>,
    /// dense id -> runtime id, or `None` for a module that is not live.
    runtime_ids: Vec<Option<usize>>,
    /// dense id -> the chunk file whose bytes carry that module's factory. A module
    /// absent here lives in the entry chunk.
    chunk_of: HashMap<DenseModuleId, String>,
    /// dense id -> `"./chunk.js"` for each dynamic-import root, which is how a
    /// render rewrites `import()` targets.
    chunk_names: HashMap<DenseModuleId, String>,
}

pub struct Bundler {
    entry: DenseModuleId,
    ids: Vec<SharedModuleId>,
    indices: HashMap<SharedModuleId, DenseModuleId>,
    resolver: Resolvers,
    resolution_cache: ResolutionCache,
    frontend_pool: ThreadPool,
    modules: Vec<Option<ModuleState>>,
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
/// ([`crate::tailwind::resolve_scans`]) over the cached parts, which reads no files and
/// tokenizes nothing. Same algorithm as a from-scratch scan, same bytes out.
struct TailwindScan {
    per_file: HashMap<PathBuf, crate::tailwind::SourceScan>,
}

impl TailwindScan {
    fn candidates(&self) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        crate::tailwind::resolve_scans(self.per_file.values(), &mut out);
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
        let resolver = Resolvers::new(config);
        // Use every core: parse/transform dominates cold-build CPU and scales
        // near-linearly (each module is independent). The old `.min(4)` cap
        // held a 32-core machine to ~2.7 CPUs utilized on a 1000-module cold
        // build — the single largest cold-wall-time cost found by profiling.
        let frontend_threads = std::thread::available_parallelism().map_or(1, usize::from);
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
                config.virtual_modules.clone(),
                config.import_meta_env.clone(),
                config.import_meta_glob.clone(),
                config.defines.clone(),
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
                config.server_external_packages.clone(),
            ),
            frontend_pool: ThreadPoolBuilder::new()
                .num_threads(frontend_threads)
                .thread_name(|index| format!("diffpack-frontend-{index}"))
                .build()
                .map_err(|error| format!("cannot create frontend worker pool: {error}"))?,
            modules: Vec::new(),
            target: config.target,
            hmr: config.hmr,
            config: config.clone(),
            render_cache: Mutex::new(RenderCache::default()),
            emit_plans: Mutex::new(HashMap::new()),
            map_root: OnceLock::new(),
            tailwind_scan_cache: Mutex::new(HashMap::new()),
            tailwind_sheet_cache: Mutex::new(Vec::new()),
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
        self.indices
            .get(id.as_ref())
            .and_then(|&index| self.modules[index].as_ref())
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
            .indices
            .get(id.as_ref())
            .copied()
            .filter(|&index| self.modules[index].is_some());
        if let Some(index) = known {
            let Some(old) = self.modules[index].clone() else {
                unreachable!("known index always holds a module");
            };
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
        self.ids
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                self.modules[*index]
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
        let Some(old) = self.modules[index].clone() else {
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
        Ok(1 + self.discover_from(new_paths, delta, diagnostics, true)?)
    }

    /// Every currently-loaded module whose loader id has the same filesystem path
    /// as `path_id` but carries a query or fragment — i.e. a virtual module
    /// derived from that physical file (a `?tsr-split=*` route chunk, a `?url`
    /// asset, a `?raw` inline). These must be re-derived when the physical file
    /// changes. Returns `(dense index, full id string)` pairs.
    fn derived_virtual_siblings(&self, path_id: &str) -> Vec<(DenseModuleId, String)> {
        self.ids
            .iter()
            .enumerate()
            .filter(|(index, id)| {
                self.modules[*index].is_some() && {
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
        stats.written.extend(write_server_runtime_entry(&server_dir, options.hmr)?);
        prune_stale_files(&server_dir, &stats.written)?;
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
        prune_stale_files(&public_dir, &stats.written)?;
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
    /// ([`Self::emit_css`]: candidate scan, Tailwind compile, concatenation in
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
            .filter_map(|id| self.indices.get(id.as_str()).copied())
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
        let (summary, written) = self.emit_web_written(reachable, output_dir, entry_file, options)?;
        prune_stale_files(output_dir, &written)?;
        Ok(summary)
    }

    /// Emit one browser page into `output_dir` (entry chunk `entry_file`, its
    /// dynamic-import chunks, extracted `<entry-stem>.css`, and content-hashed
    /// assets) WITHOUT pruning stale files, returning the summary and the exact
    /// set of files written. This is the multi-page primitive: a MULTI-PAGE build
    /// emits every page into a shared `output_dir` (page chunks named per page,
    /// assets deduped by content hash), accumulates every page's `written` set, and
    /// prunes ONCE at the end via [`prune_web_output`] — so a shared asset written
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
        stats.written.extend(write_server_runtime_entry(&server_dir, options.hmr)?);
        prune_stale_files(&server_dir, &stats.written)?;
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
        let live_stage = crate::build_profile::stage("emit/live-modules");
        let reachable = self.live_modules(reachable);
        drop(live_stage);
        let reachable_dense = reachable
            .iter()
            .filter_map(|id| self.indices.get(id.as_str()).copied())
            .collect::<Vec<_>>();
        let allowed = reachable_dense.iter().copied().collect::<HashSet<_>>();
        if cancel.cancelled() {
            stats.cancelled = true;
            return Ok(stats);
        }
        let assets_stage = crate::build_profile::stage("emit/assets");
        self.emit_assets(&allowed, parent, &mut stats.written, cancel)?;
        drop(assets_stage);
        let mut runtime_ids = vec![None; self.ids.len()];
        for (runtime_id, &dense_id) in reachable_dense.iter().enumerate() {
            runtime_ids[dense_id] = Some(runtime_id);
        }
        let main_modules = self.static_closure(self.entry, &allowed);
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
        let plan_stage = crate::build_profile::stage("emit/chunk-plan");
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
        let chunk_names = Self::chunk_names(&plans);
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
                if let Some(module) = self.modules[dense].as_ref() {
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
                    let file_name =
                        format!("{stem}-{:08x}.worker.js", content_hash(entry.to_string_lossy().as_bytes()) as u32);
                    if let Some((_, existing)) = emitted_paths.iter().find(|(path, _)| *path == entry) {
                        worker_urls.push((
                            format!("__diffpack_worker__{key}__"),
                            format!("{}assets/{existing}", self.resolution_cache.base),
                        ));
                        continue;
                    }
                    emitted_paths.push((entry.clone(), file_name.clone()));
                    WORKER_DEPTH.with(|cell| cell.set(depth + 1));
                    let result = (|| -> Result<(), String> {
                        let (worker_bundler, update) =
                            Bundler::discover_direct_with_config(&entry, &self.config)?;
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
        // Honest-format guards: refuse (naming the module and the way out)
        // rather than emit output Node rejects at parse. `import.meta` is a
        // syntax error anywhere in a CommonJS file, and a top-level `await` has
        // no CommonJS spelling either — a CJS module body is synchronous, so
        // there is nothing to lower it to. In ESM output (production AND dev/HMR)
        // it is representable: the module renders as an `async` factory and the
        // registry's `__pending` table makes every importer wait on it.
        for &dense in &reachable_dense {
            let Some(module) = self.modules[dense].as_ref() else {
                continue;
            };
            if module.uses_import_meta && !options.format.is_esm() {
                return Err(format!(
                    "`import.meta` in {} is a syntax error in CommonJS output; bundle with \
                     `--format esm` (where it resolves against the emitted chunk)",
                    self.ids[dense]
                ));
            }
            if module.uses_top_level_await && !options.format.is_esm() {
                return Err(format!(
                    "top-level await in {} cannot be represented in CommonJS output; \
                     bundle with `--format esm`",
                    self.ids[dense]
                ));
            }
        }
        // Which modules must render as `async` factories. Empty (and free) unless
        // some reachable module actually top-level-`await`s; when one does, the
        // property propagates up every static import edge, exactly as an ES
        // module's "async evaluation" does.
        let async_stage = crate::build_profile::stage("emit/async-closure");
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
                let css_stage = crate::build_profile::stage("emit/css");
                let mut written = BTreeSet::new();
                if self.emit_css(&allowed, output, &mut written, cancel)? {
                    cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                drop(css_stage);
                Ok(written)
            },
            || -> Result<(), String> {
                let split_stage = crate::build_profile::stage("emit/render-split-chunks");
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
                let main_stage = crate::build_profile::stage("emit/render-main-chunk");
                let (rendered, main_key, main_fresh) = self.render_chunk_cached(
                    &main_modules,
                    &[self.entry],
                    &chunk_names,
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
        let mut flags = vec![false; self.modules.len()];
        let mut queue: Vec<DenseModuleId> = Vec::new();
        for &dense in reachable {
            if runtime_ids[dense].is_some()
                && self.modules[dense]
                    .as_ref()
                    .is_some_and(|module| module.uses_top_level_await)
            {
                flags[dense] = true;
                queue.push(dense);
            }
        }
        if queue.is_empty() {
            return Ok(AsyncModules { flags, any: false });
        }
        // Reverse static edges, restricted to modules that are actually emitted.
        let mut importers: HashMap<DenseModuleId, Vec<(DenseModuleId, String)>> = HashMap::new();
        for &dense in reachable {
            let Some(module) = self.modules[dense].as_ref() else {
                continue;
            };
            if runtime_ids[dense].is_none() {
                continue;
            }
            for (specifier, target, _) in &module.dependencies {
                if runtime_ids[*target].is_none() || module.pruned_imports.contains(specifier) {
                    continue;
                }
                importers
                    .entry(*target)
                    .or_default()
                    .push((dense, specifier.clone()));
            }
        }
        while let Some(dense) = queue.pop() {
            let Some(edges) = importers.get(&dense) else {
                continue;
            };
            for (importer, specifier) in edges {
                let module = self.modules[*importer]
                    .as_ref()
                    .expect("an emitted importer must exist");
                match AwaitableImport::classify(&module.code, specifier) {
                    AwaitableImport::None => continue,
                    AwaitableImport::Statement | AwaitableImport::ReExportAll => {}
                    AwaitableImport::LazyNamespace => {
                        return Err(format!(
                            "{} does `export * as ... from {:?}`, and {} uses top-level await: \
                             the namespace re-export is a lazy getter, which cannot await the \
                             module's initialisation. Import it with a normal \
                             `import * as ns from {:?}; export {{ ns }}` instead",
                            self.ids[*importer], specifier, self.ids[dense], specifier
                        ));
                    }
                    AwaitableImport::BareRequire => {
                        return Err(format!(
                            "{} reaches {} through a CommonJS `require({:?})`, and that module \
                             uses top-level await: a synchronous `require` cannot wait for it \
                             (Node throws ERR_REQUIRE_ASYNC_MODULE here too). Reach it with a \
                             static `import` or a dynamic `import()` instead",
                            self.ids[*importer], self.ids[dense], specifier
                        ));
                    }
                }
                if !flags[*importer] {
                    flags[*importer] = true;
                    queue.push(*importer);
                }
            }
        }
        Ok(AsyncModules { flags, any: true })
    }

    /// DETECTION-ONLY variant of [`Self::async_module_closure`]: which reachable
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
            .filter_map(|id| self.indices.get(id.as_str()).copied())
            .collect();
        let in_reachable: HashSet<DenseModuleId> = reachable_dense.iter().copied().collect();
        let mut flags = vec![false; self.modules.len()];
        let mut queue: Vec<DenseModuleId> = Vec::new();
        for &dense in &reachable_dense {
            if self.modules[dense]
                .as_ref()
                .is_some_and(|module| module.uses_top_level_await)
            {
                flags[dense] = true;
                queue.push(dense);
            }
        }
        if queue.is_empty() {
            return HashSet::new();
        }
        let mut importers: HashMap<DenseModuleId, Vec<(DenseModuleId, String)>> = HashMap::new();
        for &dense in &reachable_dense {
            let Some(module) = self.modules[dense].as_ref() else {
                continue;
            };
            for (specifier, target, _) in &module.dependencies {
                if !in_reachable.contains(target) || module.pruned_imports.contains(specifier) {
                    continue;
                }
                importers
                    .entry(*target)
                    .or_default()
                    .push((dense, specifier.clone()));
            }
        }
        while let Some(dense) = queue.pop() {
            let Some(edges) = importers.get(&dense) else {
                continue;
            };
            for (importer, specifier) in edges {
                let Some(module) = self.modules[*importer].as_ref() else {
                    continue;
                };
                match AwaitableImport::classify(&module.code, specifier) {
                    AwaitableImport::Statement | AwaitableImport::ReExportAll => {}
                    AwaitableImport::None
                    | AwaitableImport::LazyNamespace
                    | AwaitableImport::BareRequire => continue,
                }
                if !flags[*importer] {
                    flags[*importer] = true;
                    queue.push(*importer);
                }
            }
        }
        flags
            .iter()
            .enumerate()
            .filter(|(_, flagged)| **flagged)
            .map(|(dense, _)| self.ids[dense].to_string())
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
        let best_stage = crate::build_profile::stage("emit/render-best");
        let rendered = self.render_best(
            modules,
            roots,
            chunk_names,
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
            let validate_stage = crate::build_profile::stage("emit/validate-mappings");
            self.validate_chunk_mappings(&bundle.code, &bundle.mappings, chunk_name)?;
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
                let minify_stage = crate::build_profile::stage("emit/minify");
                let (minified, minified_map) =
                    minify_chunk_code_with_map(&bundle.code, chunk_name)?;
                drop(minify_stage);
                let compose_stage = crate::build_profile::stage("emit/compose-map");
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
        let mut hasher = DefaultHasher::new();
        (format as u8).hash(&mut hasher);
        // Whether the BUILD has any async module decides the runtime's shape
        // (`__pending`, `require.esmAsync`, an awaiting entry tail), so it is
        // build-wide cache state; the per-module flags that shape THIS chunk's
        // factories are folded in with each member below.
        async_modules.any.hash(&mut hasher);
        // The HMR runtime shape (register-only guard, per-module `module.hot`,
        // `replace`/cache invalidation) changes the emitted bytes, so a dev (hmr)
        // chunk is a distinct cache entry from its production form.
        hmr.hash(&mut hasher);
        // `minify` shapes the emitted bytes (a minified chunk differs from its
        // readable form), so it is part of the cache key: a leaf edit that changes
        // one chunk re-minifies exactly that chunk and reuses the rest byte-for-byte.
        minify.hash(&mut hasher);
        // `source_map` decides whether the minify branch also composes and stores a
        // `map_json` on the cached bundle, so a source-mapped chunk is a distinct
        // cache entry from its plain-minified form (never a silent map mismatch).
        source_map.hash(&mut hasher);
        is_main.hash(&mut hasher);
        // Membership and load order are as much a part of this chunk's bytes as
        // its modules are: which roots it owns decides its export demand and its
        // tail, the prerequisite file names are literally emitted as imports at
        // the top of the file, and `flat_allowed` picks between two entirely
        // different renderers. Without these a re-partition (a route gaining a
        // shared dependency, say) would silently serve the previous partition's
        // bytes from the render cache. All three describe THIS chunk, so a leaf
        // edit elsewhere still leaves the key — and the cached bytes — untouched.
        roots.hash(&mut hasher);
        prerequisites.hash(&mut hasher);
        flat_allowed.hash(&mut hasher);
        modules.len().hash(&mut hasher);
        for &dense in modules {
            dense.hash(&mut hasher);
            match self.modules[dense].as_ref() {
                Some(module) => {
                    // Key on the TRANSFORMED-output identity, not the source hash:
                    // a chunk whose members emit byte-identical output is reused
                    // even if a member's source text changed (e.g. a route edit
                    // whose body lives in a split chunk leaves the reference module
                    // byte-identical).
                    module.code_hash.hash(&mut hasher);
                    hash_optional_id(&mut hasher, runtime_ids[dense]);
                    hash_export_demand(&mut hasher, &global_demands[dense]);
                    // An `async` factory and an awaited import site are different
                    // bytes, and the target's flag is what decides the latter — so
                    // both this module's flag and each target's are part of the key.
                    async_modules.is_async(dense).hash(&mut hasher);
                    for (specifier, target, demand) in &module.dependencies {
                        specifier.hash(&mut hasher);
                        demand.dynamic.hash(&mut hasher);
                        demand.eager.hash(&mut hasher);
                        demand.all.hash(&mut hasher);
                        demand.names.hash(&mut hasher);
                        target.hash(&mut hasher);
                        hash_optional_id(&mut hasher, runtime_ids[*target]);
                        async_modules.is_async(*target).hash(&mut hasher);
                        if demand.dynamic {
                            match chunk_names.get(target) {
                                Some(name) => {
                                    1u8.hash(&mut hasher);
                                    name.hash(&mut hasher);
                                }
                                None => 0u8.hash(&mut hasher),
                            }
                        }
                    }
                }
                // A chunk member is always present (it came from the reachable
                // static closure); encode the absent case distinctly rather than
                // panic, so a future caller cannot get a silent collision.
                None => u64::MAX.hash(&mut hasher),
            }
        }
        hasher.finish()
    }

    /// The dynamic-import roots that become their own chunks: every dynamically
    /// imported target not already in the entry's static closure, sorted by id and
    /// deduplicated. This is the single source of truth for chunk assignment, so
    /// [`Self::emit_with_options`] and [`Self::client_route_manifest`] agree on the
    /// order — and therefore the `<stem>.chunk-<n>` names — of every chunk.
    fn dynamic_roots(&self, allowed: &HashSet<DenseModuleId>) -> Vec<DenseModuleId> {
        let main_set = self
            .static_closure(self.entry, allowed)
            .into_iter()
            .collect::<HashSet<_>>();
        let mut roots = allowed
            .iter()
            .flat_map(|source| {
                self.modules[*source]
                    .iter()
                    .flat_map(|module| module.dependencies.iter())
                    .filter(|(_, _, demand)| demand.deferred())
                    .map(|(_, target, _)| *target)
            })
            .filter(|target| !main_set.contains(target))
            .collect::<Vec<_>>();
        roots.sort_by(|left, right| self.ids[*left].cmp(&self.ids[*right]));
        roots.dedup();
        roots
    }

    /// The build's chunk partition: the single source of truth every consumer
    /// (emit, the route manifest, HMR location) derives its chunk assignment from,
    /// so all three describe the same files with the same contents.
    ///
    /// The main chunk holds `static_closure(entry)` and is never represented here;
    /// its modules are excluded from every returned chunk, so nothing the entry
    /// already carries is duplicated into a split chunk. Every remaining live
    /// module is then labelled with the SET of dynamic roots that can reach it
    /// statically, and modules sharing a label become one chunk. A module reachable
    /// from a single root stays private to that root's chunk; a module two routes
    /// share is extracted into one shared chunk that both routes list as a
    /// prerequisite. This is what makes membership disjoint — the property the
    /// runtime already assumed, since `runtime_ids` are global and `__require`
    /// throws rather than re-running a second copy of a factory.
    ///
    /// Naming is deterministic because the render cache and the incremental thesis
    /// key on bytes: a chunk that owns exactly one root and nothing else keeps the
    /// historical `<stem>.chunk-<n>` name derived from that root's position in
    /// [`Self::dynamic_roots`] (whose ordering is deliberately untouched), and
    /// every other chunk is numbered `<stem>.shared-<n>` in label order.
    fn chunk_plan(
        &self,
        allowed: &HashSet<DenseModuleId>,
        entry_file: &str,
    ) -> Result<Vec<ChunkPlan>, String> {
        let (stem, extension) = split_file_name(entry_file)?;
        let main_set = self
            .static_closure(self.entry, allowed)
            .into_iter()
            .collect::<HashSet<_>>();
        let roots = self.dynamic_roots(allowed);
        let mut closures = Vec::with_capacity(roots.len());
        for (index, &root) in roots.iter().enumerate() {
            // A root outside the live set would give an empty closure, and the
            // chunk named by every rewritten `import()` of it would hold no
            // factory — `__require` would then throw "Module is not loaded" at
            // runtime. Fail the build here, where the cause is still visible.
            if !allowed.contains(&root) {
                return Err(format!(
                    "dynamic-import root {} (chunk {}) was dropped from the live module set; \
                     its chunk would be empty and importing it would fail at runtime",
                    self.ids[root],
                    index + 1
                ));
            }
            closures.push(
                self.static_closure(root, allowed)
                    .into_iter()
                    .collect::<HashSet<_>>(),
            );
        }

        // Label -> members. `BTreeMap` over the sorted root-index vector fixes both
        // the grouping and the order chunks are numbered in, independent of hash
        // iteration order, so repeated builds emit byte-identical files.
        let mut groups: BTreeMap<Vec<usize>, Vec<DenseModuleId>> = BTreeMap::new();
        let mut ordered = allowed.iter().copied().collect::<Vec<_>>();
        ordered.sort_by(|left, right| self.ids[*left].cmp(&self.ids[*right]));
        for dense in ordered {
            if main_set.contains(&dense) {
                continue;
            }
            let label = closures
                .iter()
                .enumerate()
                .filter(|(_, closure)| closure.contains(&dense))
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            // Every live module is reachable from the entry, so it lies in the
            // entry's static closure or in some dynamic root's. An empty label
            // means it belongs to no chunk and would be silently dropped from the
            // output; that is a graph bug, not something to paper over.
            if label.is_empty() {
                return Err(format!(
                    "live module {} is in neither the entry closure nor any dynamic-import \
                     closure, so no chunk would carry it",
                    self.ids[dense]
                ));
            }
            groups.entry(label).or_default().push(dense);
        }

        let mut plans = Vec::with_capacity(groups.len());
        let mut shared_count = 0_usize;
        for (label, members) in groups {
            let member_set = members.iter().copied().collect::<HashSet<_>>();
            let chunk_roots = label
                .iter()
                .copied()
                .map(|index| roots[index])
                .filter(|root| member_set.contains(root))
                .collect::<Vec<_>>();
            // The historical name survives only for the shape it described: a
            // chunk that is exactly one root's private closure. Anything else — a
            // set shared by several roots, or a single root's leftovers once the
            // root module itself was pulled into a shared chunk — is a new kind of
            // artifact and gets a name that says so.
            let file_name = match (label.as_slice(), chunk_roots.as_slice()) {
                ([index], [root]) => chunk_file_name(&stem, &extension, index + 1, self.ids[*root].as_ref()),
                _ => {
                    shared_count += 1;
                    format!("{stem}.shared-{shared_count}{extension}")
                }
            };
            plans.push(ChunkPlan {
                members,
                roots: chunk_roots,
                prerequisites: Vec::new(),
                file_name,
            });
        }

        // Dense id -> owning chunk index; `None` is the main chunk (or a module
        // outside the live set), which is always already loaded.
        let mut chunk_of = vec![None::<usize>; self.ids.len()];
        for (index, plan) in plans.iter().enumerate() {
            for &member in &plan.members {
                chunk_of[member] = Some(index);
            }
        }
        let mut edges = Vec::with_capacity(plans.len());
        for (index, plan) in plans.iter().enumerate() {
            let mut prerequisites = Vec::new();
            for &member in &plan.members {
                let Some(module) = self.modules[member].as_ref() else {
                    continue;
                };
                for (_, target, demand) in &module.dependencies {
                    if demand.deferred() || !allowed.contains(target) {
                        continue;
                    }
                    // A target in the main chunk needs no prerequisite: the main
                    // chunk is what installs the runtime, so it has always been
                    // evaluated before any split chunk loads.
                    if let Some(other) = chunk_of[*target]
                        && other != index
                    {
                        prerequisites.push(other);
                    }
                }
            }
            prerequisites.sort_unstable();
            prerequisites.dedup();
            edges.push(prerequisites);
        }
        // Only DIRECT edges are recorded. A prerequisite's own prerequisites are
        // imported by that chunk's own header, and ESM/CJS both finish evaluating
        // an import before the importer's body runs, so the direct edges close
        // transitively at load time.
        for (plan, prerequisites) in plans.iter_mut().zip(edges) {
            plan.prerequisites = prerequisites;
        }
        Ok(plans)
    }

    /// `dense id -> "./<chunk file>"` for every dynamic-import root, naming the
    /// chunk that actually CONTAINS the root — which may be a shared chunk, since a
    /// root reachable from another root is extracted like any other shared module.
    fn chunk_names(plans: &[ChunkPlan]) -> HashMap<DenseModuleId, String> {
        let mut names = HashMap::new();
        for plan in plans {
            for &root in &plan.roots {
                names.insert(root, format!("./{}", plan.file_name));
            }
        }
        names
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
    pub fn client_route_manifest(
        &self,
        reachable: &BTreeSet<ModuleId>,
        entry_file: &str,
        base: &str,
    ) -> Result<crate::manifest::ClientRouteManifest, String> {
        // The manifest must describe the SAME chunk set emit produces, so refine
        // the reachable set through the identical dead-module elimination pass and
        // then read the chunk assignment off the identical plan.
        let reachable = self.live_modules(reachable);
        let allowed = reachable
            .iter()
            .filter_map(|id| self.indices.get(id.as_str()).copied())
            .collect::<HashSet<_>>();
        let plans = self.chunk_plan(&allowed, entry_file)?;
        let mut routes: BTreeMap<String, Vec<String>> = BTreeMap::new();
        routes.insert(
            crate::manifest::ROOT_ROUTE_ID.to_string(),
            vec![entry_file.to_string()],
        );
        for (index, plan) in plans.iter().enumerate() {
            for &root in &plan.roots {
                let Some(route_id) = split_chunk_route_id(self.ids[root].as_ref())? else {
                    continue;
                };
                // Preloading the route's own chunk is no longer enough: its shared
                // dependencies live in prerequisite chunks, and a browser that
                // fetched only the route chunk would register factories whose
                // dependencies are missing. List the prerequisite closure ahead of
                // the chunk itself, in the order the runtime needs them.
                let preloads = routes.entry(route_id).or_default();
                let mut ordered = Vec::new();
                let mut seen = HashSet::new();
                chunk_load_order(&plans, index, &mut seen, &mut ordered);
                for file in ordered {
                    if !preloads.contains(&file) {
                        preloads.push(file);
                    }
                }
            }
        }
        Ok(crate::manifest::ClientRouteManifest {
            base: base.to_string(),
            entry: entry_file.to_string(),
            routes,
        })
    }

    /// Derives the client build's client-references manifest (Manifest #1 —
    /// `bundlerConfig`; see docs/RSC_SPEC.md §1) from the SAME chunk partition emit
    /// produces, so every `id`/`chunks` value describes the bytes on disk.
    ///
    /// Every reachable `"use client"` module becomes one entry keyed by its
    /// `module_reference_id` (canonical path) — the exact string the react-server
    /// build's `$$id` prefix carries, so the flight reference and this manifest
    /// agree on the module. `id` is the module's numeric runtime id (what
    /// `__webpack_require__` takes over diffpack's registry); `chunks` is `[]` when
    /// the module lands in the already-loaded main entry chunk, otherwise its single
    /// hosting split chunk `[chunkFile, chunkFile]` (file name doubles as chunk id);
    /// `name` is `"*"` (the real export arrives via the `$$id` split).
    pub fn client_references_manifest(
        &self,
        reachable: &BTreeSet<ModuleId>,
        entry_file: &str,
    ) -> Result<crate::rsc::ClientReferencesManifest, String> {
        let reachable = self.live_modules(reachable);
        let reachable_dense = reachable
            .iter()
            .filter_map(|id| self.indices.get(id.as_str()).copied())
            .collect::<Vec<_>>();
        let allowed = reachable_dense.iter().copied().collect::<HashSet<_>>();
        let mut runtime_ids = vec![None; self.ids.len()];
        for (runtime_id, &dense) in reachable_dense.iter().enumerate() {
            runtime_ids[dense] = Some(runtime_id);
        }
        // The same partition emit used, so a module's `chunks` names the chunk whose
        // bytes actually carry its factory. A module not in any split chunk is in the
        // main entry chunk (already loaded), so its `chunks` is empty.
        let plans = self.chunk_plan(&allowed, entry_file)?;
        let mut chunk_of: HashMap<DenseModuleId, &str> = HashMap::new();
        for plan in &plans {
            for &member in &plan.members {
                chunk_of.insert(member, plan.file_name.as_str());
            }
        }
        let mut entries = BTreeMap::new();
        for &dense in &reachable_dense {
            let Some(module) = self.modules[dense].as_ref() else {
                continue;
            };
            let path = Path::new(self.ids[dense].as_ref());
            if crate::rsc::detect_directive(path, module.source.as_ref())
                != Some(crate::rsc::RscDirective::Client)
            {
                continue;
            }
            let id = runtime_ids[dense]
                .expect("a reachable module has a runtime id");
            let chunks = match chunk_of.get(&dense) {
                // The file name doubles as the chunk id (docs/RSC_SPEC.md §1); the
                // seam's `__diffpack_chunkFilenames` maps that id to its public URL.
                Some(file) => vec![(*file).to_string(), (*file).to_string()],
                None => Vec::new(),
            };
            entries.insert(
                crate::rsc::module_reference_id(path),
                crate::rsc::ClientReferenceEntry {
                    id,
                    chunks,
                    name: "*".to_string(),
                },
            );
        }
        Ok(crate::rsc::ClientReferencesManifest { entries })
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
    fn emit_plan(&self, reachable: &BTreeSet<ModuleId>, entry_file: &str) -> Result<Arc<EmitPlan>, String> {
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
            .filter_map(|id| self.indices.get(id.as_str()).copied())
            .collect::<Vec<_>>();
        let allowed = reachable_dense.iter().copied().collect::<HashSet<_>>();
        let plans = self.chunk_plan(&allowed, entry_file)?;
        let plan = Arc::new(self.build_emit_plan(reachable_dense, allowed, &plans));
        self.record_emit_plan(entry_file, Arc::clone(&plan))?;
        Ok(plan)
    }

    /// Assemble an [`EmitPlan`] from the pieces an emit already computed, so the
    /// stored plan is literally the one that rendered the bundle now on disk.
    fn build_emit_plan(
        &self,
        reachable_dense: Vec<DenseModuleId>,
        allowed: HashSet<DenseModuleId>,
        plans: &[ChunkPlan],
    ) -> EmitPlan {
        let mut runtime_ids = vec![None; self.ids.len()];
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
            chunk_names: Self::chunk_names(plans),
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
            let Some(&dense) = self.indices.get(module_id.as_str()) else {
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
    /// goes beside it under the shared [`Self::source_map_sidecar`] naming.
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
                format!("cannot write hmr source map {}: {error}", map_path.display())
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
            .filter_map(|id| self.indices.get(id.as_str()).copied())
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
            let Some(module) = self.modules[dense].as_ref() else {
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
                    let _stage = crate::build_profile::stage("emit/assets-image-variants");
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
                        let target_h = ((intrinsic_h as u64 * width as u64)
                            / intrinsic_w as u64)
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
        let candidates_stage = crate::build_profile::stage("css/tailwind-candidate-scan");
        // What the scan actually depends on: where it walks and what it skips. The
        // entry's remaining text (rules, theme, `@plugin`) does not enter it, so an
        // edit to the sheet keeps the scan — which is the whole point, since that walk
        // is the expensive half of compiling a monorepo's stylesheet.
        let globs = tailwind_source_globs(css)?;
        let key = (scan_root.clone(), out_root.to_path_buf(), format!("{globs:?}"));
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
        let theme_stage = crate::build_profile::stage("css/tailwind-app-theme");
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
        let _compile_stage = crate::build_profile::stage("css/tailwind-compile");
        match crate::tailwind::native_gap(css, app_theme.as_deref()) {
            Some(gap) => {
                let Some(sheet) =
                    crate::tailwind_delegate::compile(css_path, css, candidates, &gap, cancel)?
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
                        crate::tailwind::VERSION,
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
                        crate::tailwind::VERSION
                    ),
                );
                crate::tailwind::compile_with_theme_lenient(css, candidates, app_theme.as_deref())
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
    ) -> Result<Option<HashMap<PathBuf, crate::tailwind::SourceScan>>, String> {
        let mut skip = ScanSkip::for_root(scan_root, out_root);
        let (included, excluded) = globs;
        skip.excluded = excluded
            .iter()
            .flat_map(|pattern| expand_braces(pattern))
            .map(|pattern| path_segments(Path::new(&pattern)))
            .collect();
        let read_stage = crate::build_profile::stage("css/tailwind-read-sources");
        let mut sources = Vec::new();
        // The walk reads every source file under the root — hundreds of milliseconds on
        // a monorepo — so it asks between directories whether it is still wanted.
        if !collect_scan_sources(scan_root, &mut sources, &skip, &cancel) {
            return Ok(None);
        }
        for pattern in included {
            collect_glob_sources(pattern, &mut sources, &skip);
            if cancel.cancelled() {
                return Ok(None);
            }
        }
        drop(read_stage);
        let scan_stage = crate::build_profile::stage("css/tailwind-scan-candidates");
        // Tokenized per file and KEPT per file: an edit then re-tokenizes one file
        // instead of the tree (see [`TailwindScan`]). Batched with a yield point
        // between batches, because tokenizing thousands of files is ~150 ms.
        let mut per_file = HashMap::new();
        for batch in sources.chunks(128) {
            if cancel.cancelled() {
                return Ok(None);
            }
            for (path, source) in batch {
                per_file.insert(path.clone(), crate::tailwind::scan_source_parts(source));
            }
        }
        drop(scan_stage);
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
                    scan.per_file
                        .insert(path.to_path_buf(), crate::tailwind::scan_source_parts(&source));
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
        let order = self.static_execution_order(self.entry, allowed).unwrap_or_else(|| {
            let mut ids = allowed.iter().copied().collect::<Vec<_>>();
            ids.sort_by(|left, right| self.ids[*left].cmp(&self.ids[*right]));
            ids
        });
        // Remote `@import`s (scheme URLs that cannot be inlined) are hoisted,
        // deduped, to the very top: an @import is only valid before all rules,
        // so leaving one at its source position in the concatenation would be
        // silently ignored by the browser.
        let mut stylesheet = String::new();
        let mut hoisted = BTreeSet::new();
        for dense in &order {
            if let Some(module) = self.modules[*dense].as_ref() {
                for external in &module.css_external_imports {
                    if hoisted.insert(external.clone()) {
                        stylesheet.push_str(external);
                        stylesheet.push('\n');
                    }
                }
            }
        }
        for dense in order {
            if let Some(module) = self.modules[dense].as_ref()
                && let Some(css) = &module.css
            {
                if !stylesheet.is_empty() && !stylesheet.ends_with('\n') {
                    stylesheet.push('\n');
                }
                // A globally-imported Tailwind v4 entry carries its RAW source;
                // compile it here against freshly-scanned class candidates,
                // exactly as the `?url` asset path does in `emit_assets`.
                if crate::tailwind::needs_native_tailwind_compile(css) {
                    let css_path = Path::new(self.ids[dense].as_ref());
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
                        .filter(|(_, _, demand)| !demand.deferred())
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
                            if !shared.already_loaded(target)
                                && guard.0.insert(target.clone())
                            {
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
                self.indices
                    .get(path.as_ref())
                    .is_none_or(|index| self.modules[*index].is_none())
            })
            .collect::<BTreeSet<_>>();
        let shared = DiscoverShared {
            resolver: &self.resolver,
            resolution_cache: &self.resolution_cache,
            target: self.target,
            hmr: self.hmr,
            source_maps: self.config.source_maps,
            indices: &self.indices,
            modules: &self.modules,
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
                    dependencies.push((specifier, target_index, demand));
                }
                self.modules[source] = Some(ModuleState {
                    hash: loaded.hash,
                    code_hash: loaded.code_hash,
                    dependencies,
                    pruned_imports: loaded.pruned_imports,
                    source: loaded.source,
                    flat_module: loaded.flat_module,
                    code: loaded.code,
                    assets: loaded.assets,
                    css: loaded.css,
                    css_source_files: loaded.css_source_files,
                    css_external_imports: loaded.css_external_imports,
                    externals: loaded.externals,
                    droppable: loaded.droppable,
                    liveness: loaded.liveness,
                    uses_top_level_await: loaded.uses_top_level_await,
                    uses_import_meta: loaded.uses_import_meta,
                    uses_cjs_globals: loaded.uses_cjs_globals,
                    uses_dirname: loaded.uses_dirname,
                    workers: loaded.workers,
                    map: loaded.map,
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
        diagnostics: &mut Vec<Diagnostic>,
    ) -> Result<ModuleState, String> {
        let id = module_id(path);
        // A build-generated virtual module (its source is not on disk) claims this
        // id first.
        if let Some(source) = self.resolution_cache.virtual_module_source(&id) {
            let special = synthesize_virtual_module(source)?;
            let resolved = resolve_special_dependencies(
                &self.resolver,
                &self.resolution_cache,
                &id,
                self.target,
                &special,
                diagnostics,
            );
            let dependencies = resolved
                .dependencies
                .into_iter()
                .map(|(specifier, target, demand)| (specifier, self.intern(target), demand))
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
                css: special.css,
                css_source_files: special.css_source_files,
                css_external_imports: special.css_external_imports,
                externals: resolved.externals,
                droppable: false,
                liveness: ModuleLiveness::default(),
                uses_top_level_await: false,
                uses_import_meta: false,
                uses_cjs_globals: false,
                uses_dirname: false,
                workers: Vec::new(),
                map: None,
            });
        }
        // A loader (query, stylesheet, or asset) may claim this id before it is
        // ever read as JavaScript.
        if let Some(special) = load_special_module(&id, path, self.target, &self.resolution_cache) {
            let mut special = special?;
            // DEV-ONLY: a `?tsr-split=<component>` module is the extracted route
            // component — a React Fast Refresh boundary. Append the self-accept
            // footer (keyed by the split id, stable across edits and unique per
            // split) so an edit swaps the component type while preserving state.
            // `build-app` never sets `hmr`, so production splits are unaffected.
            if self.hmr
                && self.target == Target::Client
                && crate::hmr::is_refresh_boundary(
                    Path::new(id.as_ref()),
                    &[],
                    "",
                    self.resolution_cache.jsx_extensions,
                )
            {
                special
                    .code
                    .push_str(&crate::hmr::fast_refresh_footer(id.as_ref()));
            }
            let resolved = resolve_special_dependencies(
                &self.resolver,
                &self.resolution_cache,
                &id,
                self.target,
                &special,
                diagnostics,
            );
            let dependencies = resolved
                .dependencies
                .into_iter()
                .map(|(specifier, target, demand)| (specifier, self.intern(target), demand))
                .collect();
            // A `?worker` module bundles its referenced entry as a self-contained
            // worker chunk. The key matches the `__diffpack_worker__<key>__`
            // placeholder the synthesizer emitted, so the emit-step substitution
            // (shared with the `new Worker(new URL(...))` path) resolves it to the
            // emitted URL.
            let resource = ResourceId::parse(&id);
            let workers = if resource.loader_kind() == Some(LoaderKind::Worker) {
                vec![(worker_import_key(&resource.path), PathBuf::from(&resource.path))]
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
                css: special.css,
                css_source_files: special.css_source_files,
                css_external_imports: special.css_external_imports,
                externals: resolved.externals,
                droppable: false,
                liveness: ModuleLiveness::default(),
                uses_top_level_await: false,
                uses_import_meta: false,
                uses_cjs_globals: false,
                uses_dirname: false,
                workers,
                map: None,
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
        // A `.vue`/`.svelte` component is compiled to JavaScript by the app's own
        // compiler FIRST; everything below then treats the result as the module's
        // source, so the component's imports become graph edges and its
        // TypeScript is stripped by the ordinary transform.
        let (component_code, language, component) =
            match precompile_component(path, &source, &self.resolution_cache)? {
                Some(compiled) => (
                    Some(compiled.code),
                    compiled.language,
                    compiled.side_effects,
                ),
                None => (
                    None,
                    crate::transform::SourceLanguage::FromPath,
                    ComponentSideEffects::default(),
                ),
            };
        let module_text = component_code.as_deref().unwrap_or(source.as_str());
        let source = self
            .resolution_cache
            .apply_vite_replacements(path, module_text, self.target)?;
        let project_config = project_config_for(&self.resolver, path, &self.resolution_cache.jsx)?;
        let mut transformed = crate::transform::transform_module_in_language(
            path,
            &source,
            self.target,
            self.hmr && self.target == Target::Client,
            self.resolution_cache.jsx_extensions,
            &project_config,
            language,
            self.config.source_maps,
        );
        // The text the map's positions were measured against is `source`, which
        // is the file on disk ONLY when neither a component compiler nor Vite's
        // replacements rewrote it. When one did, the map says so, so nothing can
        // read its positions as offsets into the file.
        mark_rewritten_source(
            &mut transformed,
            component_code.is_some(),
            matches!(source, Cow::Owned(_)),
        );
        diagnostics.extend(source_diagnostics(path, &transformed.diagnostics));

        // DEV-ONLY: install `import.meta.hot` -> `module.hot`, and append the React
        // Fast Refresh self-accept footer to client component modules. Gated on the
        // bundler's HMR flag, which `build-app` never sets, so production output is
        // untouched. Runs on the lowered factory body, where `module`/`exports` and
        // the runtime-installed `module.hot` are in scope.
        if self.hmr {
            let before_refresh = std::mem::take(&mut transformed.code);
            let hot_rewritten = before_refresh.contains("import.meta.hot");
            transformed.code = crate::hmr::rewrite_import_meta_hot(&before_refresh, self.target);
            let mut preamble_lines = 0_u32;
            if self.target == Target::Client {
                let module_key = path.to_string_lossy();
                // Namespace the Fast Refresh family ids by module BEFORE anything
                // reads them (see `fast_refresh_preamble`). Every instrumented
                // module needs this, boundary or not: the collisions that wedge the
                // browser come from the registrations, not from the accept wiring.
                if crate::hmr::needs_fast_refresh_preamble(&transformed.code) {
                    let preamble = crate::hmr::fast_refresh_preamble(&module_key);
                    preamble_lines =
                        preamble.bytes().filter(|byte| *byte == b'\n').count() as u32;
                    transformed.code.insert_str(0, &preamble);
                }
                if crate::hmr::is_refresh_boundary(
                    path,
                    &transformed.liveness.exports,
                    &source,
                    self.resolution_cache.jsx_extensions,
                ) {
                    transformed
                        .code
                        .push_str(&crate::hmr::fast_refresh_footer(&module_key));
                }
            }
            rebase_map_for_refresh(
                &mut transformed.map,
                &before_refresh,
                preamble_lines,
                hot_rewritten,
            );
        }

        // A component's `<style>` `@import`s are graph edges of the component
        // module, exactly as a stylesheet's own `@import`s are of that stylesheet.
        // Borrowed (no copy) for every ordinary module, which is all of them but
        // the components.
        let (dependency_specifiers, dependency_demands) =
            component_dependencies(&transformed, &component);
        let resolved_dependencies = resolve_dependencies(
            &self.resolver,
            &self.resolution_cache,
            path,
            self.target,
            &dependency_specifiers,
            &dependency_demands,
            diagnostics,
        );
        let dependencies = resolved_dependencies
            .dependencies
            .into_iter()
            .map(|(specifier, target, demand)| (specifier, self.intern(target), demand))
            .collect();

        let code_hash = content_hash(transformed.code.as_bytes());
        let droppable = module_droppable(path, diagnostics);
        Ok(ModuleState {
            hash,
            code_hash,
            dependencies,
            pruned_imports: resolved_dependencies.pruned_imports,
            source: Arc::from(source),
            flat_module: transformed.flat_module,
            code: transformed.code,
            assets: component.assets,
            css: component.css,
            css_source_files: component.css_source_files,
            css_external_imports: component.css_external_imports,
            externals: resolved_dependencies.externals,
            droppable,
            liveness: transformed.liveness,
            uses_top_level_await: transformed.uses_top_level_await,
            uses_import_meta: transformed.uses_import_meta,
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
            && let Some(flat) =
                self.render_flat(reachable, roots, chunk_names, global_demands, is_main, format)
        {
            return Ok(Some(flat));
        }
        // A top-level `await` that reaches the registry runtime is rendered as an
        // `async` factory (see `render_runtime`); `async_module_closure` has already
        // refused every import site that cannot carry the matching `await`.
        Ok(self.render_runtime(
            reachable,
            roots,
            chunk_names,
            runtime_ids,
            global_demands,
            prerequisites,
            is_main,
            async_modules,
            format,
            hmr,
            cancel,
        ))
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
        // A flat chunk's public surface is a single `export{...}` list taken from
        // one module's flat exports, so it can only speak for a chunk with exactly
        // one dynamic root. A shared chunk (no root) or a chunk two roots landed in
        // has no such surface; those render through the registry runtime, which
        // addresses modules by runtime id and needs no exports at all.
        let entry = match (is_main, roots) {
            (true, _) => self.entry,
            (false, [root]) => *root,
            (false, _) => return None,
        };
        let reachable_set = reachable.iter().copied().collect::<HashSet<_>>();
        for &dense_index in reachable {
            let module = self.modules[dense_index].as_ref()?;
            module.flat_module.as_ref()?;
            // The flat path strips import bindings and cannot bind an external's
            // `require`. A module with externals renders through the runtime path.
            if !module.externals.is_empty() {
                return None;
            }
            // A free `exports`/`module`/`require` reference needs the CJS-style
            // factory wrapper that defines those names. Concatenated at the top
            // level of an ESM chunk it would throw `exports is not defined`, so
            // such a module renders through the runtime path there. (In CJS
            // output the host wrapper provides them, so flat stays fine.)
            if format.is_esm() && module.uses_cjs_globals {
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
                if !demand.deferred()
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
                map_json: None,
            });
        }
        let order = self.static_execution_order(entry, &included)?;
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

        // Start from the global (cross-chunk) demand so a module keeps every
        // export any chunk imports from it. The chunk's own root is an entry
        // point: for a dynamic chunk its full namespace can be read after
        // `import()`, so it keeps all exports; the main entry's public surface
        // is provided by the outer wrapper, so it demands nothing here.
        let mut demands = global_demands.to_vec();
        if is_main {
            demands[entry] = ExportDemand::default();
        } else {
            demands[entry].all = true;
        }
        // Per-module shake + dynamic-import rewrite is independent work; run it
        // across the pool (order preserved by the indexed collect) and keep only
        // the concatenation serial. On a 1000-module cold build the serial
        // version of this loop was a measurable slice of total wall time.
        let shaken = self.frontend_pool.install(|| {
            order
                .par_iter()
                .map(|&dense_index| -> Option<(String, Option<LineTrack>)> {
                    let module = self.modules[dense_index].as_ref()?;
                    let flat = module.flat_module.as_ref()?;
                    // Track lines only when this module actually has a real map to
                    // carry; otherwise the bookkeeping would be pure cost.
                    let wanted = module.map.is_some() && flat.map_lines.is_some();
                    let (mut module_code, shake_lines) = shake_module_code(
                        &flat.code,
                        &demands[dense_index],
                        &module.pruned_imports,
                        wanted,
                    );
                    // shaken lines -> flat lines -> the module map's lines.
                    let mut track = match (shake_lines, flat.map_lines.as_ref()) {
                        (Some(shake), Some(flat_lines)) => Some(shake.compose(flat_lines)),
                        _ => None,
                    };
                    for (specifier, target, demand) in &module.dependencies {
                        if !demand.dynamic {
                            continue;
                        }
                        let chunk = chunk_names.get(target)?;
                        let import = format!("import({})", quote(specifier));
                        let lowered_import = format!("__dynamic(require, {})", quote(specifier));
                        // In ESM output the split chunk is a real `.mjs`, so a
                        // native dynamic `import()` of the chunk path loads it
                        // and resolves to its namespace. In CJS output the chunk
                        // is `module.exports`, so the load goes through the host
                        // `require`.
                        let replacement = if format.is_esm() {
                            format!("import({})", quote(chunk))
                        } else {
                            format!("Promise.resolve().then(()=>require({}))", quote(chunk))
                        };
                        // The specifier rewrite is an in-place edit of a real line
                        // of user code, so it is recorded column-exactly: a token
                        // inside the old specifier is dropped, one after it moves
                        // by the difference in width.
                        let (needle, present) = if module_code.contains(&import) {
                            (import, true)
                        } else if module_code.contains(&lowered_import) {
                            (lowered_import, true)
                        } else {
                            (String::new(), false)
                        };
                        if !present {
                            return None;
                        }
                        module_code = match track.as_mut() {
                            Some(track) => {
                                let mut edits = LineTrack::identity(module_code.lines().count());
                                let rewritten = crate::source_map::replace_tracked(
                                    &module_code,
                                    &needle,
                                    &replacement,
                                    &mut edits,
                                )
                                .unwrap_or_else(|| module_code.clone());
                                *track = edits.compose(track);
                                rewritten
                            }
                            None => module_code.replace(&needle, &replacement),
                        };
                    }
                    Some((module_code, track))
                })
                .collect::<Vec<_>>()
        });
        let mut code = String::new();
        let mut mappings = Vec::with_capacity(order.len());
        let mut generated_line = 0_u32;
        for (&dense_index, shaken) in order.iter().zip(&shaken) {
            let (module_code, track) = shaken.as_ref()?;
            if !module_code.is_empty() {
                let generated_lines = module_code.lines().count() as u32;
                mappings.push(ModuleMapping {
                    dense_index,
                    generated_line,
                    tokens: self.project_module_tokens(dense_index, track.as_ref(), generated_line),
                });
                generated_line += generated_lines;
                code.push_str(module_code);
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
            // A flat chunk only ever has clean named exports (the flat builder
            // bails to the runtime path on any `default`/re-export/`export *`),
            // so a bare `export{a,b}` of the in-scope bindings is always valid
            // ESM; the CJS chunk exposes the same names on `module.exports`.
            if format.is_esm() {
                code.push_str(&format!("export{{{}}};\n", exports.join(",")));
            } else {
                code.push_str(&format!("module.exports={{{}}};\n", exports.join(",")));
            }
        }
        // The browser entry runs first, so its process/NODE_ENV shim must precede
        // any module code (and any later dynamically-imported chunk). The runtime
        // path injects this via the entry prelude; the flat path prepends it here.
        if is_main && format == ModuleFormat::BrowserEsm && self.config.browser_process_shim {
            code.insert_str(0, BROWSER_GLOBALS_PRELUDE);
            for mapping in &mut mappings {
                mapping.generated_line += 1;
                for token in &mut mapping.tokens {
                    token.generated_line += 1;
                }
            }
        }
        Some(RenderedBundle {
            code,
            mappings,
            map_json: None,
        })
    }

    /// Moves one module's REAL map tokens onto the chunk's generated lines.
    /// `track` says which line of the module map's text each line of the module's
    /// emitted region came from (and what was rewritten inside it); everything it
    /// could not account for contributes nothing, so those positions stay
    /// UNMAPPED rather than being given a plausible origin.
    fn project_module_tokens(
        &self,
        dense_index: DenseModuleId,
        track: Option<&LineTrack>,
        generated_line: u32,
    ) -> Vec<MapToken> {
        let Some(track) = track else {
            return Vec::new();
        };
        let Some(map) = self.modules[dense_index]
            .as_ref()
            .and_then(|module| module.map.as_ref())
        else {
            return Vec::new();
        };
        let mut tokens = Vec::new();
        track.project(map, generated_line, &mut tokens);
        tokens
    }

    fn static_execution_order(
        &self,
        root: DenseModuleId,
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
                if !demand.deferred() && allowed.contains(target) {
                    visit(bundler, *target, allowed, states, order)?;
                }
            }
            states.insert(source, 2);
            order.push(source);
            Some(())
        }

        let mut order = Vec::with_capacity(allowed.len());
        // Execution order is defined by the graph's ROOT: its import statements
        // run first-to-last, depth-first. Seeding the walk from the root (not
        // from every module in id order) is what makes two sibling subtrees
        // execute in import order — `import './b'; import './a'` runs `b`
        // before `a`, and a stylesheet imported by the entry is emitted before
        // one imported by a later component, so the CSS cascade tie-breaks the
        // way the source says. Modules the root cannot reach statically (other
        // dynamic chunk roots in the allowed set) follow in stable id order.
        let mut roots = Vec::with_capacity(allowed.len());
        if allowed.contains(&root) {
            roots.push(root);
        }
        let mut rest = allowed
            .iter()
            .copied()
            .filter(|dense| *dense != root)
            .collect::<Vec<_>>();
        rest.sort_by(|left, right| self.ids[*left].cmp(&self.ids[*right]));
        roots.extend(rest);
        let mut states = HashMap::new();
        for root in roots {
            visit(self, root, allowed, &mut states, &mut order)?;
        }
        (order.len() == allowed.len()).then_some(order)
    }

    /// The RSC `__webpack_*` seam prelude for the browser entry, or `None` when the
    /// build hosts no `"use client"` module (so a non-RSC bundle is byte-identical).
    ///
    /// Installs, over diffpack's own module registry, the three globals the pinned
    /// `react-server-dom-webpack/client.browser` reads at module-init:
    /// `__webpack_require__` (→ `runtime.require(id)`), `__webpack_require__.u` /
    /// `__webpack_get_script_filename__` (chunk id → public URL), and
    /// `__webpack_chunk_load__` (id → `import(url)`; the imported chunk
    /// self-registers its factories). `__diffpack_chunkFilenames` maps every split
    /// chunk's id (its file name) to its public URL. The runtime is read LAZILY
    /// inside `__webpack_require__` (it is built by the module-graph IIFE that runs
    /// after this prelude), and an unknown chunk id is a HARD ERROR, never a silent
    /// fallback.
    fn rsc_webpack_seam(
        &self,
        reachable: &[DenseModuleId],
        chunk_names: &HashMap<DenseModuleId, String>,
    ) -> Option<String> {
        let has_client_boundary = reachable.iter().any(|&dense| {
            self.modules[dense].as_ref().is_some_and(|module| {
                crate::rsc::detect_directive(
                    Path::new(self.ids[dense].as_ref()),
                    module.source.as_ref(),
                ) == Some(crate::rsc::RscDirective::Client)
            })
        });
        if !has_client_boundary {
            return None;
        }
        // Chunk id -> public URL. The chunk's file name doubles as its id (the same
        // id `client_references_manifest` records), and the URL is `base + file`.
        let base = self.config.base.trim_end_matches('/');
        let mut chunk_filenames: BTreeMap<String, String> = BTreeMap::new();
        for name in chunk_names.values() {
            let file = name.trim_start_matches("./");
            chunk_filenames
                .entry(file.to_string())
                .or_insert_with(|| format!("{base}/{file}"));
        }
        let chunk_filenames_json = serde_json::to_string(&chunk_filenames)
            .unwrap_or_else(|_| "{}".to_string());
        let runtime_key = quote(&format!(
            "__diffpack_runtime:{}",
            self.ids[self.entry].as_ref()
        ));
        Some(format!(
            r#"globalThis.__diffpack_chunkFilenames=Object.assign(globalThis.__diffpack_chunkFilenames||{{}},{chunk_filenames_json});
(function(){{
var __rtKey={runtime_key};
function __webpack_require__(id){{var rt=globalThis[__rtKey];if(!rt)throw new Error("diffpack rsc seam: runtime "+__rtKey+" is not initialized");return rt.require(id);}}
__webpack_require__.u=function(c){{return globalThis.__diffpack_chunkFilenames[c];}};
globalThis.__webpack_require__=__webpack_require__;
globalThis.__webpack_get_script_filename__=__webpack_require__.u;
globalThis.__webpack_chunk_load__=function(c){{var f=globalThis.__diffpack_chunkFilenames[c];if(f===undefined)throw new Error("__webpack_chunk_load__: unknown chunk id "+c);return import(f);}};
}})();
"#
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn render_runtime(
        &self,
        reachable: &[DenseModuleId],
        roots: &[DenseModuleId],
        chunk_names: &HashMap<DenseModuleId, String>,
        runtime_ids: &[Option<usize>],
        global_demands: &[ExportDemand],
        prerequisites: &[String],
        is_main: bool,
        async_modules: &AsyncModules,
        format: ModuleFormat,
        hmr: bool,
        cancel: EmitCancel<'_>,
    ) -> Option<RenderedBundle> {
        // See `render_flat`: demand is aggregated globally across chunks. Each of
        // this chunk's entry points keeps its full namespace (the main entry is
        // required by the outer wrapper; a dynamic root is read as a namespace
        // after `import()`). A shared chunk has no roots and is described entirely
        // by the global demand of the chunks that import from it.
        let mut export_demands = global_demands.to_vec();
        for &root in roots {
            export_demands[root].all = true;
        }
        // The per-module render fan-out is where a big chunk spends its time, so it
        // is also where an abandoned pass has to notice: each module checks the
        // signal first, and once one has seen it every remaining module returns
        // immediately, so the whole render unwinds in a millisecond or two instead of
        // holding the dev loop for the length of an 18 MB entry chunk.
        let stop = std::sync::atomic::AtomicBool::new(false);
        let fragments = reachable
            .par_iter()
            .filter_map(|&dense_index| {
                if stop.load(std::sync::atomic::Ordering::Relaxed) {
                    return None;
                }
                if cancel.cancelled() {
                    stop.store(true, std::sync::atomic::Ordering::Relaxed);
                    return None;
                }
                let module = self.modules[dense_index].as_ref()?;
                let runtime_id = runtime_ids[dense_index]
                    .expect("a rendered module must have a deterministic runtime ID");
                // A dependency the dead-module elimination pass dropped is no
                // longer in the emitted set (no runtime id). This module was kept
                // only because OTHER exports of it are live; it places no body-use
                // demand on the dropped target, so every reference to it is a
                // re-export getter the export demand already shakes away. Strip its
                // `require(...)` line too (as a pruned import) and omit it from the
                // require map, so the emitted module never references a module that
                // was dropped from the graph.
                let mut pruned_imports = module.pruned_imports.clone();
                let mut dropped_targets: Vec<&str> = Vec::new();
                for (specifier, target, _) in &module.dependencies {
                    if runtime_ids[*target].is_none() {
                        pruned_imports.insert(specifier.clone());
                        dropped_targets.push(specifier.as_str());
                    }
                }
                // An ASYNC module (it top-level-`await`s, or transitively imports
                // something that does) renders as an `async` factory, and each of
                // its import sites that names an async target becomes an `await`
                // so this module's body does not run until that target's own
                // top-level `await` has settled — the ES module spec's async
                // evaluation ordering, expressed inside the registry runtime.
                //
                // This runs BEFORE the shake, because the shake consumes the
                // `/*__diffpack_import:...__*/` markers the rewrite anchors on.
                // The rewrite preserves each marker and adds no line, so the shake
                // (and the source map it feeds) sees exactly the same structure.
                let is_async_module = async_modules.is_async(dense_index);
                // Track lines only when this module has a real map to carry.
                let mut track = module
                    .map
                    .as_ref()
                    .map(|_| LineTrack::identity(module.code.lines().count()));
                let mut lowered = Cow::Borrowed(module.code.as_str());
                // `export * from "./x"` where `./x` was dropped by dead-module
                // elimination. Its import STATEMENT is pruned above (the marker
                // makes it addressable), but the star re-export lowers to a plain
                // runtime call — nothing marks it, so it survived the shake and
                // asked the registry for a module the bundle no longer contains.
                // The registry's miss path is the EXTERNAL path (that is how
                // `node:fs` and an uninstalled optional dependency work), so the
                // call became a raw `require("./x")` in the emitted file and threw
                // MODULE_NOT_FOUND at load. A module is only droppable when it has
                // no side effects and nothing demands an export of it, so
                // re-exporting its names re-exports nothing: the statement is
                // deleted IN PLACE, leaving the line (and so every following line,
                // and the source map) exactly where it was.
                for specifier in &dropped_targets {
                    let call = format!("__reExport(exports,require.esm({}));", quote(specifier));
                    if lowered.contains(&call) {
                        let rewritten = lowered.replace(&call, "");
                        // Both rewrites here only touch bundler-synthesized request
                        // lines and never move a line. Any line they DID change is
                        // dropped from the map rather than assumed to still describe
                        // the same text.
                        if let Some(track) = track.as_mut() {
                            track.invalidate_changed_lines(&lowered, &rewritten);
                        }
                        lowered = Cow::Owned(rewritten);
                    }
                }
                if is_async_module {
                    for (specifier, target, _) in &module.dependencies {
                        if runtime_ids[*target].is_some() && async_modules.is_async(*target) {
                            let rewritten = await_async_imports(&lowered, specifier);
                            if let Some(track) = track.as_mut() {
                                track.invalidate_changed_lines(&lowered, &rewritten);
                            }
                            lowered = Cow::Owned(rewritten);
                        }
                    }
                }
                let (code, shake_lines) = shake_module_code(
                    &lowered,
                    &export_demands[dense_index],
                    &pruned_imports,
                    track.is_some(),
                );
                let track = match (shake_lines, track) {
                    (Some(shake), Some(track)) => Some(shake.compose(&track)),
                    _ => None,
                };
                // `__dirname`/`__filename` in a BROWSER bundle. A Node bundle gets
                // them from the emitted file's own location (see the ESM prelude);
                // a browser has no such location, so — exactly as webpack does for
                // `target: "web"`, whose `node.__dirname`/`node.__filename` "mock"
                // defaults are the literals `"/"` and `"/index.js"` — the factory
                // binds them per module. Bundled CommonJS packages read them at
                // module init (Next's ncc-compiled `url`/`querystring` polyfills do
                // `__nccwpck_require__.ab = __dirname + "/"`), and without a binding
                // that is a `ReferenceError` that kills the whole entry. Emitted
                // only for modules that actually reference one, and on the factory's
                // OWN line so the module body's line numbers (and its source map)
                // are unchanged.
                let browser_cjs_locations =
                    if format == ModuleFormat::BrowserEsm && module.uses_dirname {
                        "const __filename=\"/index.js\",__dirname=\"/\";"
                    } else {
                        ""
                    };
                let module_fragment = format!(
                    "{runtime_id}:{}function(module,exports,require,__toESM,__export,__reExport,__import,__dynamic,__esmNamespace,__seal){{{browser_cjs_locations}\n{}\n}},\n",
                    if is_async_module { "async " } else { "" },
                    code
                );
                let mut map_fragment = format!("{runtime_id}:{{");
                let mut chunk_fragment = format!("{runtime_id}:{{");
                for (specifier, target, demand) in &module.dependencies {
                    let Some(target_runtime_id) = runtime_ids[*target] else {
                        // Dropped by dead-module elimination — not emitted.
                        continue;
                    };
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
                    track,
                ))
            })
            .collect::<Vec<_>>();
        // Abandoned part-way: the fragment set is incomplete, so no bundle can be
        // built from it. The caller writes nothing and keeps its debt.
        if stop.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        // A split chunk's members can statically depend on modules that landed in
        // a SIBLING chunk (shared code extracted out of both). Those factories
        // must already be registered before this chunk's are used, so the chunk
        // loads its prerequisites from its own header: ESM evaluates every import
        // before the module body, and the CJS `require` runs before the registry
        // literal is built. Only DIRECT prerequisites are listed — each of them
        // loads its own in turn, so the closure is covered transitively. The main
        // chunk has none: its members form a static closure, and it is what
        // installs the runtime the others register into.
        let prerequisite_loads = prerequisites
            .iter()
            .map(|file| {
                if format.is_esm() {
                    format!("import {};\n", quote(file))
                } else {
                    format!("require({});\n", quote(file))
                }
            })
            .collect::<String>();
        let mut prelude = match format {
            ModuleFormat::Esm => {
                // `__dirname`/`__filename` are CommonJS globals absent in an ES module,
                // but bundled CJS modules (e.g. Next's ncc-compiled internals doing
                // `__nccwpck_require__.ab = __dirname + "/"`, or Prisma's generated
                // client) still reference them. Define them from `import.meta.url` at the
                // top of EVERY Node ESM chunk, not only the entry: a chunk is its own ES
                // module, so it does NOT close over the entry's bindings, and a CJS module
                // that lands in a split chunk would throw `ReferenceError: __dirname is
                // not defined in ES module scope` the moment that chunk is imported. (That
                // is exactly how cal.com's `pages/api/**` routes died once they were
                // bundled into the SSR graph, where Prisma's generated client is reachable
                // ONLY through their lazily-imported chunks.) Every chunk is emitted into
                // the same directory as the entry, so the value each computes is the same
                // one the entry computes.
                let mut prelude = "import { fileURLToPath as __diffpackFileURLToPath } from \"node:url\";\nimport { dirname as __diffpackDirname } from \"node:path\";\nconst __filename = __diffpackFileURLToPath(import.meta.url);\nconst __dirname = __diffpackDirname(__filename);\n".to_string();
                if is_main {
                    // `createStartHandler` reads `process.env.TSS_SERVER_FN_BASE` at
                    // module-init time and caches it as the prefix it matches
                    // server-function requests against, so the default must be in
                    // place before any bundled module evaluates. This runs at the very
                    // top of the entry, before the module-graph IIFE. It is a `??=`
                    // default (never clobbers a real value) and a harmless no-op for a
                    // non-TanStack Node bundle.
                    // `createRequire` backs the entry's `requireNative` (host `require`
                    // for native addons); only the entry installs the runtime, so only
                    // the entry needs it.
                    prelude.insert_str(
                        0,
                        "import { createRequire as __diffpackCreateRequire } from \"node:module\";\n",
                    );
                    prelude.push_str("process.env.TSS_SERVER_FN_BASE ??= \"/_serverFn/\";\n");
                }
                prelude
            }
            ModuleFormat::BrowserEsm if is_main && self.config.browser_process_shim => {
                BROWSER_GLOBALS_PRELUDE.to_string()
            }
            _ => String::new(),
        };
        // RSC `__webpack_*` seam (docs/RSC_SPEC.md §1): when the CLIENT build hosts
        // any `"use client"` module, the browser entry installs the webpack globals
        // `react-server-dom-webpack/client.browser` reads at module-init, mapped
        // onto diffpack's registry. Gated strictly on a real client boundary so a
        // non-RSC browser bundle stays byte-identical. Appended to (not replacing)
        // the prelude so the process shim still runs first.
        if is_main && format == ModuleFormat::BrowserEsm
            && let Some(seam) = self.rsc_webpack_seam(reachable, chunk_names) {
                prelude.push_str(&seam);
            }
        // Lines the chunk emits before `const __newModules={`: the format's
        // prelude and one `import`/`require` per prerequisite chunk. The module
        // fragments start after them, so the source map's generated lines must be
        // shifted by exactly this much or every mapped position is off by the
        // header's height.
        let header_lines = prelude.matches('\n').count() as u32
            + prerequisite_loads.matches('\n').count() as u32;
        let mut modules = String::new();
        let mut maps = String::new();
        let mut chunks = String::new();
        let mut mappings = Vec::with_capacity(fragments.len());
        let mut module_lines = 0_u32;
        for (dense_index, module, map, chunk, generated_lines, track) in fragments {
            // `const __newModules={` shares its line with the first fragment, and
            // each fragment opens with its factory header (`<id>:function(...){`)
            // and puts the module's own first line on the next one — so
            // `3 + header_lines + module_lines` is already the line the module's
            // CODE starts on, which is where its tokens belong.
            let region_line = 3 + header_lines + module_lines;
            let _ = generated_lines;
            mappings.push(ModuleMapping {
                dense_index,
                generated_line: region_line,
                tokens: self.project_module_tokens(dense_index, track.as_ref(), region_line),
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
        // Only the main chunk names a module in its tail (it evaluates the entry
        // and returns its exports); a split chunk only registers factories, so it
        // has no single "entry" to identify.
        let entry_runtime_id = runtime_ids[self.entry]
            .expect("the entry module must have a deterministic runtime ID");
        // In ESM output (Node or browser) a split chunk is a real module, loaded
        // for its REGISTRATION side effect: `require.dynamic` imports the file and
        // then resolves the requested module by runtime id out of the shared
        // registry, so one chunk can carry several roots (and shared code that is
        // nobody's root) without any of them having to be its default export.
        // Node ESM resolves external Node built-ins through
        // `createRequire`. Browser ESM has no `node:module`; `requireNative`
        // returns a load-safe throw-on-USE stub instead: dead server code that
        // leaked into the client graph via isomorphic imports may still `require`
        // a Node built-in and read a shape off it (or `new` it) at module init,
        // so the stub lets property reads and construction succeed (the module
        // LOADS), but throws a clear, specifically-named error the moment that
        // dead code actually CALLS into the built-in — it never fabricates a
        // value. Protocol probes (`then`/`Symbol.toPrimitive`/iterators) return
        // `undefined` so the stub is neither mistaken for a thenable nor silently
        // coerced. In CJS output both go through the host `require`, as before.
        // With an async module in the build, a dynamically imported target may be
        // one; `import()` already yields a promise, so `require.dynamic` resolves
        // through `__requireAsync` and the awaited namespace is fully initialised.
        // `__chunkQuery` (defined in the runtime prelude below) is the query the
        // ENTRY chunk's own module URL carries, propagated to every chunk it
        // dynamically imports. It is empty for a normal load; it matters when a
        // host deliberately re-imports the entry under a fresh URL to get a FRESH
        // module graph — which the react-server `serve` worker does
        // (`import(entry + "?v=" + mtime)` after dropping the runtime globals). The
        // registry lives on `globalThis`, so without propagating the query the new
        // entry instance builds a new registry while every already-imported chunk
        // stays in Node's ESM cache and never re-runs its `__register`, and the
        // first `import()` of one of those chunks resolves instantly to a module
        // that registered into the DISCARDED runtime — `__require` then throws
        // "Module is not loaded: <id>".
        let require_dynamic_esm = if async_modules.any {
            r#"require.dynamic=specifier=>{const chunk=__chunks[id][specifier];if(chunk===undefined)return require(specifier);if(chunk[0]!==null)return import(chunk[0]+__chunkQuery).then(()=>__requireAsync(chunk[1]));return __requireAsync(chunk[1]);};"#
        } else {
            r#"require.dynamic=specifier=>{const chunk=__chunks[id][specifier];if(chunk===undefined)return require(specifier);if(chunk[0]!==null)return import(chunk[0]+__chunkQuery).then(()=>__require(chunk[1]));return __require(chunk[1]);};"#
        };
        let (require_dynamic, require_native) = match format {
            ModuleFormat::Esm => (
                require_dynamic_esm,
                "const requireNative=__diffpackCreateRequire(import.meta.url);".to_string(),
            ),
            ModuleFormat::BrowserEsm => (require_dynamic_esm, browser_require_native()),
            ModuleFormat::Cjs => (
                r#"require.dynamic=specifier=>{const chunk=__chunks[id][specifier];if(chunk===undefined)return require(specifier);if(chunk[0]!==null){if(typeof requireNative!=="function")throw new Error("Dynamic chunks require a CommonJS host");requireNative(chunk[0]);}return __require(chunk[1]);};"#,
                r#"const requireNative=typeof require==="function"?require:null;"#.to_string(),
            ),
        };
        // DEV-ONLY HMR wiring (never emitted for production `build-app`). A
        // version-aware dynamic import re-fetches a re-emitted chunk; `module.hot`
        // is installed per module; the runtime gains apply/invalidate methods and a
        // register-only guard; the ESM (Node) main chunk also starts a control
        // endpoint so the dev server hot-reloads the server in-process.
        let require_dynamic = if hmr && format.is_esm() {
            if async_modules.any {
                crate::hmr::REQUIRE_DYNAMIC_ESM_HMR_ASYNC
            } else {
                crate::hmr::REQUIRE_DYNAMIC_ESM_HMR
            }
        } else {
            require_dynamic
        };
        let hot_install = if hmr { "module.hot=__makeHot(id);" } else { "" };
        // The entry-id declaration + HMR methods are emitted only in HMR builds
        // (never in production `build-app` output). `__hmrPending` is how the
        // update paths in `RUNTIME_METHODS` reach the ASYNC-module bookkeeping:
        // in a build with a top-level `await` it reads the real `__pending`
        // table, and in one without it is a constant `undefined`, so a hot
        // update stays exactly as synchronous as it was.
        let hmr_methods = if hmr {
            let pending_lookup = if async_modules.any {
                "function __hmrPending(id){return __pending[id];}"
            } else {
                "function __hmrPending(){return undefined;}"
            };
            format!(
                "const __entryId={entry_runtime_id};{pending_lookup}{}",
                crate::hmr::RUNTIME_METHODS
            )
        } else {
            String::new()
        };
        let runtime_return = if hmr {
            // A chunk whose root is async returns `__runtime.requireAsync(...)`
            // (see `require_entry`), so an HMR build with an async module must
            // publish it alongside the update methods.
            let require_async_export = if async_modules.any {
                "requireAsync:__requireAsync,"
            } else {
                ""
            };
            // `register` is WRAPPED in an HMR build so newly loaded chunks invalidate
            // the reverse-import index the update walk reads (see `RUNTIME_METHODS`).
            // Every registration goes through this exported entry point — the main
            // chunk's own bootstrap and every split chunk's tail alike — so the index
            // can never be consulted stale.
            format!(
                "const __exports={{register:(m,p,c)=>{{__hmrInvalidateImporterIndex();__register(m,p,c);}},require:__require,{require_async_export}replace:__replace,hmrApply:__hmrApply,serverInvalidate:__hmrServerInvalidate,bumpVersion:__bumpVersion,prune:__hmrPrune}};globalThis[{}]=__exports;return __exports;",
                quote(crate::hmr::RUNTIME_GLOBAL)
            )
        } else if async_modules.any {
            "return {register:__register,require:__require,requireAsync:__requireAsync};"
                .to_string()
        } else {
            "return {register:__register,require:__require};".to_string()
        };
        // ASYNC MODULES. A module that top-level-`await`s renders as an `async`
        // factory, so calling it returns a promise instead of running to
        // completion. `__pending[id]` holds that promise until the module has
        // finished initialising; a rejected one is deliberately LEFT in place so a
        // later importer rejects too rather than reading a half-built namespace.
        //
        // `__require` stays synchronous and keeps returning `module.exports` — the
        // namespace object is created (and its getters installed) by the factory's
        // synchronous prefix, before its first `await`, so its identity is stable
        // and every existing caller is unaffected. Waiting is the caller's job, via
        // `require.esmAsync`/`require.async` (emitted at an async module's own
        // import sites) or `__requireAsync` (the chunk tail and `require.dynamic`).
        // Both first run `require`, which is what populates `__pending`, and only
        // then look the pending promise up.
        //
        // Every line here is emitted only when the build actually has an async
        // module, so an ordinary bundle's runtime is byte-for-byte what it was.
        let (require_async, require_async_runtime, factory_call) = if async_modules.any {
            (
                "  require.async=specifier=>{const target=__maps[id][specifier],value=require(specifier),pending=target===undefined?undefined:__pending[target];return pending?pending.then(()=>value):value;};\n  require.esmAsync=specifier=>{const target=__maps[id][specifier],namespace=require.esm(specifier),pending=target===undefined?undefined:__pending[target];return pending?pending.then(()=>namespace):namespace;};\n",
                "const __pending=Object.create(null);\nfunction __requireAsync(id){const exports=__require(id),pending=__pending[id];return pending?pending.then(()=>exports):exports;}\n",
                "const __result=factory(module,module.exports,require,__toESM,__export,__reExport,__import,__dynamic,__esmNamespace,__seal);\n  if(__result&&typeof __result.then===\"function\")__pending[id]=__result.then(()=>{delete __pending[id];});",
            )
        } else {
            (
                "",
                "",
                "factory(module,module.exports,require,__toESM,__export,__reExport,__import,__dynamic,__esmNamespace,__seal);",
            )
        };
        let reimport_guard = if hmr { crate::hmr::REIMPORT_GUARD } else { "" };
        let server_control = if hmr && format == ModuleFormat::Esm {
            crate::hmr::SERVER_CONTROL
        } else {
            ""
        };
        // `__toESM` decides whether a required module is ALREADY an ES namespace or a
        // CommonJS `module.exports` that needs interop. That decision is made on
        // `__esmNamespaces`, a brand only `__esmNamespace` (i.e. only diffpack's own
        // ESM emit) can add to, plus a null-prototype `Symbol.toStringTag === "Module"`
        // test for a namespace the HOST produced. It is deliberately NOT made on
        // `__esModule`: that is a convention marker any CommonJS file may stamp on its
        // own `exports` — tslib's UMD build and every TypeScript package published with
        // `importHelpers` do — and treating it as proof of ESM handed such a module
        // straight through, so `import x from "tslib"` threw "does not provide an export
        // named default". Node's ESM-imports-CJS rule ignores `__esModule` entirely:
        // `default` is `module.exports`, which is what the interop below builds.
        //
        // Three properties the interop must hold, each of which was a defect:
        //
        //  * IDEMPOTENT AND STABLE. `__toESM` is not called once per module but once
        //    per import site, and `export * as ns from "cjs"` re-runs it on every
        //    read of `ns`. `__cjsNamespaces` keys the wrapper by the `module.exports`
        //    it wraps, so one CommonJS module has exactly one namespace object (as in
        //    Node), and `__isESM` recognises a wrapper (via `__cjsInterops`) so
        //    re-wrapping a wrapper is a no-op instead of nesting `default.default`.
        //    Keying by `module.exports` cannot cover `module.exports = 42` — a
        //    WeakMap takes no primitive key — so a static import goes through
        //    `require.esm`, which keys by the MODULE ID (`__idNamespaces`) and is the
        //    only identity that exists for every value shape. Both halves matter and
        //    neither alone is enough: id-keying alone would give two modules that
        //    each `module.exports = 42` one shared namespace under a value-keyed
        //    cache, and exports-keying alone gives ONE module a fresh namespace per
        //    read (`ns.legacy === ns.legacy` was `false` against Node's `true`).
        //  * STRICT ABOUT NAMED EXPORTS. `import { missing } from "./legacy.cjs"` is a
        //    hard error in Node; it must not evaluate to `undefined` here. The wrapper
        //    is therefore NOT exempt from `__import`'s check — the check consults the
        //    live `module.exports` and throws when the name is on neither.
        //  * LIVE, NOT A SNAPSHOT. The wrapper's enumerable keys are copied from
        //    `module.exports` at wrap time, which in an ESM<->CJS cycle is a
        //    PARTIALLY populated object. `__syncCJS` re-copies on every later
        //    `__toESM` of the same exports, and `__import` reads through to the live
        //    `module.exports`, so a key the module assigns after the wrap is visible
        //    rather than permanently missing.
        // Whether the module this chunk EVALUATES is async, and so whether the
        // chunk's own wrapper has to await it. `requireAsync` returns a promise
        // only for a module still initialising, so the wrapper's `await` is what
        // makes the chunk's default export the module's FINISHED namespace.
        let awaits_evaluation = if is_main {
            async_modules.is_async(self.entry)
        } else {
            matches!(roots, [only] if async_modules.is_async(*only))
        };
        let require_entry = if awaits_evaluation {
            "requireAsync"
        } else {
            "require"
        };
        // See `require_dynamic_esm`: the entry chunk's own query, re-attached to
        // every chunk URL so one runtime instance only ever loads chunk instances
        // that register into it. Empty (and therefore inert) unless the host
        // imported the entry with a query. CommonJS output has no `import.meta`
        // and no such protocol, so it declares nothing.
        let chunk_query = if format.is_esm() {
            "const __chunkQuery=(()=>{const __q=import.meta.url.indexOf(\"?\");return __q<0?\"\":import.meta.url.slice(__q);})();\n"
        } else {
            ""
        };
        let tail = if is_main {
            format!(
                r#"const __runtime=globalThis[{runtime_key}]??=(()=>{{
const __modules=Object.create(null),__maps=Object.create(null),__chunks=Object.create(null),__cache=Object.create(null);
{chunk_query}const __exportStates=new WeakMap(),__esmNamespaces=new WeakSet(),__cjsNamespaces=new WeakMap(),__cjsInterops=new WeakMap(),__cjsOrigins=new WeakMap(),__idNamespaces=Object.create(null);
function __esmNamespace(){{const namespace=Object.create(null);Object.defineProperty(namespace,Symbol.toStringTag,{{value:"Module"}});__esmNamespaces.add(namespace);return namespace;}}
function __seal(namespace){{const movable=Reflect.ownKeys(namespace).filter(key=>typeof key==="string"&&Object.getOwnPropertyDescriptor(namespace,key).configurable);const sorted=[...movable].sort();if(movable.some((key,index)=>key!==sorted[index])){{const descriptors={{}};for(const key of movable){{descriptors[key]=Object.getOwnPropertyDescriptor(namespace,key);delete namespace[key];}}for(const key of sorted)Object.defineProperty(namespace,key,descriptors[key]);}}for(const key of Reflect.ownKeys(namespace)){{const descriptor=Object.getOwnPropertyDescriptor(namespace,key);if(descriptor?.configurable)Object.defineProperty(namespace,key,{{configurable:false}});}}Object.preventExtensions(namespace);}}
function __exportState(target){{let state=__exportStates.get(target);if(!state){{state={{explicit:new Set(),stars:new Map(),ambiguous:new Set()}};__exportStates.set(target,state);}}return state;}}
function __export(target,name,getter){{const state=__exportState(target);const descriptor=Object.getOwnPropertyDescriptor(target,name);if(descriptor?.configurable)delete target[name];if(!Object.prototype.hasOwnProperty.call(target,name))Object.defineProperty(target,name,{{enumerable:true,configurable:true,get:getter}});state.explicit.add(name);state.stars.delete(name);state.ambiguous.delete(name);}}
function __reExport(target,source){{const state=__exportState(target);for(const key of Object.keys(source)){{if(key==="default"||key==="__esModule"||state.explicit.has(key)||state.ambiguous.has(key))continue;const previous=state.stars.get(key);if(previous&&previous!==source){{delete target[key];state.stars.delete(key);state.ambiguous.add(key);continue;}}if(!previous){{Object.defineProperty(target,key,{{enumerable:true,configurable:true,get:()=>source[key]}});state.stars.set(key,source);}}}}}}
function __holdsProperties(value){{return value!==null&&value!==undefined&&(typeof value==="object"||typeof value==="function");}}
function __origin(exports,specifier){{if(__holdsProperties(exports)&&!__cjsOrigins.has(exports))__cjsOrigins.set(exports,specifier);return exports;}}
function __isESM(value){{if(!value||(typeof value!=="object"&&typeof value!=="function"))return false;if(__esmNamespaces.has(value)||__cjsInterops.has(value))return true;return Object.getPrototypeOf(value)===null&&value[Symbol.toStringTag]==="Module";}}
function __syncCJS(namespace,value){{if(__holdsProperties(value))for(const key of Object.keys(value))if(key!=="default"&&!Object.prototype.hasOwnProperty.call(namespace,key))__export(namespace,key,()=>value[key]);return namespace;}}
function __transpiledESM(value){{
  if(!__holdsProperties(value)||!Object.prototype.hasOwnProperty.call(value,"default"))return false;
  try{{return !!value.__esModule;}}catch{{return false;}}
}}
function __toESM(value){{
  if(__isESM(value))return value;
  const cached=__cjsNamespaces.get(value);
  if(cached)return __syncCJS(cached,value);
  const namespace=Object.create(null);
  Object.defineProperty(namespace,"__esModule",{{value:true}});
  // The `__esModule` interop. A CommonJS module that BOTH stamps `__esModule` and owns
  // a `default` property was compiled down from ESM (TypeScript / Babel / SWC output —
  // which is most of npm), so its `default` IS the module's default export and
  // `import X from` must bind that function, not the exports object wrapping it. Every
  // other bundler does this (esbuild's `__toESM`, Rollup/Vite's `interopDefault`,
  // webpack's `_interop_require_default`), and it is the whole reason the marker
  // exists; without it `import CredentialsProvider from "next-auth/providers/
  // credentials"` binds `{{__esModule:true,default:fn}}` and calling it throws
  // "is not a function" — which is exactly how cal.com's next-auth config died.
  //
  // Without a `default` property the marker says nothing about what a default import
  // should be, so the Node rule stands: the default export is `module.exports`.
  if(__transpiledESM(value))__export(namespace,"default",()=>value.default);
  else __export(namespace,"default",()=>value);
  __syncCJS(namespace,value);
  __cjsInterops.set(namespace,{{exports:value}});
  if(__holdsProperties(value))__cjsNamespaces.set(value,namespace);
  return namespace;
}}
function __namespaceOf(id,value){{
  if(__holdsProperties(value))return __toESM(value);
  const cached=__idNamespaces[id];
  if(cached)return cached;
  const namespace=__toESM(value);
  __idNamespaces[id]=namespace;
  return namespace;
}}
function __import(namespace,name){{
  if(Object.prototype.hasOwnProperty.call(namespace,name))return namespace[name];
  const interop=__cjsInterops.get(namespace);
  if(interop&&__holdsProperties(interop.exports)&&Object.prototype.hasOwnProperty.call(interop.exports,name)){{const exports=interop.exports;__export(namespace,name,()=>exports[name]);return exports[name];}}
  const origin=__cjsOrigins.get(namespace)??(interop?__cjsOrigins.get(interop.exports):undefined);
  throw new SyntaxError("The requested module "+(origin===undefined?"(unknown)":JSON.stringify(origin))+" does not provide an export named "+JSON.stringify(name));
}}
function __dynamic(require,specifier){{return Promise.resolve().then(()=>require.dynamic(specifier)).then(exports=>__toESM(__origin(exports,specifier)));}}
function __register(modules,maps,chunks){{Object.assign(__modules,modules);Object.assign(__maps,maps);Object.assign(__chunks,chunks);}}
function __require(id){{
  if(__cache[id])return __cache[id].exports;
  const factory=__modules[id];
  if(!factory)throw new Error("Module is not loaded: "+id);
  const module={{exports:{{}}}};
  __cache[id]=module;
  {hot_install}
  const require=specifier=>{{const target=__maps[id][specifier];if(target===undefined){{if(requireNative)return __origin(requireNative(specifier),specifier);throw new Error("Cannot resolve "+specifier+" from "+id);}}return __origin(__require(target),specifier);}};
  require.esm=specifier=>{{const target=__maps[id][specifier],value=require(specifier);return target===undefined?__toESM(value):__namespaceOf(target,value);}};
{require_async}  {require_dynamic}
  {factory_call}
  return module.exports;
}}
{require_async_runtime}{require_native}
{hmr_methods}
{runtime_return}
}})();
{server_control}
__runtime.register(__newModules,__newMaps,__newChunks);
{reimport_guard}
return __runtime.{require_entry}({entry_runtime_id});"#
            )
        } else {
            // A split chunk always REGISTERS; whether it also evaluates depends on
            // how it can be consumed, and there are two ways.
            //
            // `require.dynamic` evaluates the requested module by runtime id, so
            // registration alone is enough for it. But a chunk is ALSO imported
            // directly as an ES module: the generated SSR router does
            // `import manifest from "./_tanstack-start-manifest_v.mjs"` and reads
            // the factory off the default export. That consumer needs the default
            // export to BE the root's namespace, so a chunk with exactly one root
            // evaluates it and returns its exports.
            //
            // A chunk with several roots, or a purely shared chunk with none, has
            // no single namespace that could stand for it. Nothing imports those
            // directly (only `require.dynamic` and prerequisite headers name them),
            // and evaluating a root the caller did not ask for would run its side
            // effects early, so they register and return the runtime.
            let evaluate = match roots {
                [only] => {
                    let root_runtime_id = runtime_ids[*only]
                        .expect("a chunk root must have a deterministic runtime ID");
                    format!("return __runtime.{require_entry}({root_runtime_id});")
                }
                _ => "return __runtime;".to_string(),
            };
            format!(
                r#"const __runtime=globalThis[{runtime_key}];
if(!__runtime)throw new Error("Diffpack runtime is not initialized");
__runtime.register(__newModules,__newMaps,__newChunks);
{reimport_guard}
{evaluate}"#
            )
        };
        // The registry runtime is identical across formats; only the module
        // boundary differs. CJS assigns the entry's exports to `module.exports`.
        // Both ESM variants bind them to a local and re-export as the default. The
        // Node ESM main chunk (which builds the runtime) imports `createRequire`
        // so external Node built-ins resolve — each emitted `.mjs` then truly
        // executes under Node's ESM goal, not merely passing `node --check`. The
        // browser ESM main chunk omits that import (a browser cannot resolve
        // `node:module`), so the entry loads and runs in the browser.
        // A chunk that evaluates an ASYNC module can only publish its finished
        // namespace by awaiting it, so its wrapper becomes an async IIFE and the
        // chunk itself top-level-`await`s — legal in an ES module (which is why
        // `emit_with_options` still refuses top-level await in CommonJS output),
        // and it makes the emitted file's own evaluation async exactly as the
        // source module graph's is.
        let (open_wrapper, close_wrapper) = if awaits_evaluation {
            ("await (async()=>{", "})()")
        } else {
            ("(()=>{", "})()")
        };
        let code = if format.is_esm() {
            format!(
                r#"{prelude}{prerequisite_loads}const __diffpackEntry={open_wrapper}
"use strict";
const __newModules={{{modules}}};
const __newMaps={{{maps}}};
const __newChunks={{{chunks}}};
{tail}
{close_wrapper};
export default __diffpackEntry;
"#
            )
        } else {
            format!(
                r#"module.exports=(()=>{{
"use strict";
{prerequisite_loads}const __newModules={{{modules}}};
const __newMaps={{{maps}}};
const __newChunks={{{chunks}}};
{tail}
}})();
"#
            )
        };
        Some(RenderedBundle {
            code,
            mappings,
            map_json: None,
        })
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
        // Labels are owned here so they outlive the borrowing builder.
        let labels: HashMap<DenseModuleId, String> = mappings
            .iter()
            .map(|mapping| {
                let origin = self.modules[mapping.dense_index]
                    .as_ref()
                    .and_then(|module| module.map.as_ref())
                    .map_or(MapOrigin::File, ModuleSourceMap::origin);
                (
                    mapping.dense_index,
                    self.source_label(mapping.dense_index, root, origin),
                )
            })
            .collect();
        // Every token in the chunk, in generated order, each remembering which
        // module it belongs to — so the walk below can visit the chunk's lines in
        // order and see, for each one, whether anything maps it.
        let mut ordered: Vec<(&MapToken, DenseModuleId)> = mappings
            .iter()
            .flat_map(|mapping| {
                mapping
                    .tokens
                    .iter()
                    .map(move |token| (token, mapping.dense_index))
            })
            .collect();
        ordered.sort_by_key(|(token, _)| (token.generated_line, token.generated_column));
        let mut builder = SourceMapBuilder::default();
        builder.set_file(output_name);
        // A module joins `sources` the first time one of its tokens is emitted, so
        // the map lists only the sources the emitted bytes really came from.
        let mut source_ids: HashMap<DenseModuleId, u32> = HashMap::new();
        let mut index = 0_usize;
        for line in 0..line_count(code) {
            let start = index;
            while index < ordered.len() && ordered[index].0.generated_line == line {
                index += 1;
            }
            // Whether this line ends up with any real mapping. Tokens can survive
            // to here and still resolve into no module (a module dropped after its
            // tokens were projected), and a line left with nothing on it is a line
            // that inherits the previous line's origin — so the marker is decided
            // on what was actually emitted, not on whether tokens were present.
            let mut mapped = false;
            for (token, dense) in &ordered[start..index] {
                let Some(module) = self.modules[*dense].as_ref() else {
                    continue;
                };
                let Some(map) = module.map.as_ref() else {
                    continue;
                };
                let Some(label) = labels.get(dense) else {
                    continue;
                };
                mapped = true;
                let source_id = match source_ids.get(dense) {
                    Some(id) => *id,
                    None => {
                        let id = builder
                            .add_source_and_content(label.as_str(), map.source_text(&module.source));
                        source_ids.insert(*dense, id);
                        id
                    }
                };
                let name = token
                    .name
                    .and_then(|index| map.names().get(index as usize))
                    .filter(|name| is_identifier(name))
                    .map(|name| builder.add_name(name.as_str()));
                builder.add_token(
                    token.generated_line,
                    token.generated_column,
                    token.source_line,
                    token.source_column,
                    Some(source_id),
                    name,
                );
            }
            if !mapped {
                // Nothing on this line came from a module: say so, rather than let
                // it inherit the last mapping before it. Nothing was emitted for
                // the line, so the marker is still the line's first segment.
                builder.add_token(line, 0, 0, 0, None, None);
            }
        }
        builder.into_sourcemap().to_json_string()
    }

    /// Refuses a chunk whose readable mappings do not describe its bytes.
    ///
    /// Every token here claims "this exact generated position came from that exact
    /// source position". A position that is not IN the emitted text — a line past
    /// the end of the chunk, a column past the end of its line — cannot have come
    /// from anywhere, so it is proof that some stage's bookkeeping drifted from the
    /// text it was tracking. That is exactly the failure this whole path exists to
    /// prevent, and it is silent in the emitted map (a consumer just resolves the
    /// wrong thing), so it is checked here and fails the build naming the chunk
    /// rather than shipping.
    ///
    /// Checked on the READABLE bytes, which is the one place both output shapes
    /// pass through: a minified chunk's map is composed from these same tokens.
    fn validate_chunk_mappings(
        &self,
        code: &str,
        mappings: &[ModuleMapping],
        chunk_name: &str,
    ) -> Result<(), String> {
        let lines: Vec<u32> = code.lines().map(crate::source_map::utf16_len).collect();
        for mapping in mappings {
            for token in &mapping.tokens {
                let module = self.ids[mapping.dense_index].as_ref();
                let Some(&width) = lines.get(token.generated_line as usize) else {
                    return Err(format!(
                        "source map for chunk `{chunk_name}` puts a token from `{module}` on \
                         generated line {}, but the chunk has only {} lines",
                        token.generated_line,
                        lines.len()
                    ));
                };
                if token.generated_column > width {
                    return Err(format!(
                        "source map for chunk `{chunk_name}` puts a token from `{module}` at \
                         generated {}:{}, but that line is {width} columns wide",
                        token.generated_line, token.generated_column
                    ));
                }
            }
        }
        Ok(())
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
                let origin = self.modules[mapping.dense_index]
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

        let mut builder = SourceMapBuilder::default();
        builder.set_file(output_name);
        let mut source_ids: HashMap<DenseModuleId, u32> = HashMap::new();
        let mut mapped_any = false;
        // Resolution (the binary search per minified token, plus the occasional
        // name verification against the module's source text) is read-only work
        // over millions of tokens for a large chunk, so it runs across the pool in
        // generated-order slices. Emission stays serial below: source ids and
        // names must be assigned in first-use order for the output to stay
        // byte-identical to the single-threaded composition.
        let tokens: Vec<oxc_sourcemap::Token> = minified_map.get_tokens().collect();
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
                                self.resolve_minified_token(
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
                    self.resolve_minified_token(
                        minified,
                        minified_map,
                        &readable,
                        &mut hint,
                        &mut source_lines,
                    )
                })
                .collect()
        };
        for (minified, resolved) in tokens.iter().zip(resolved) {
            // A token that resolved to no module at all is marked as having no
            // origin. Emitted in generated order alongside the real tokens, so the
            // map stays sorted.
            let Some(token) = resolved else {
                builder.add_token(minified.get_dst_line(), minified.get_dst_col(), 0, 0, None, None);
                continue;
            };
            let dense = token.dense;
            let source_id = match source_ids.get(&dense) {
                Some(id) => *id,
                None => {
                    let (Some(module), Some(label)) =
                        (self.modules[dense].as_ref(), labels.get(&dense))
                    else {
                        // The resolver only returns a module it verified has a
                        // map and a label, so this cannot happen; refuse loudly
                        // rather than emit a token pointing at a missing source.
                        return Err(format!(
                            "source-map composition for chunk `{chunk_name}` resolved a token \
                             into module {dense}, which has no map or label"
                        ));
                    };
                    let Some(map) = module.map.as_ref() else {
                        return Err(format!(
                            "source-map composition for chunk `{chunk_name}` resolved a token \
                             into module {dense}, which has no module map"
                        ));
                    };
                    let id = builder
                        .add_source_and_content(label.as_str(), map.source_text(&module.source));
                    source_ids.insert(dense, id);
                    id
                }
            };
            let name = token.name.map(|name| builder.add_name(name));
            builder.add_token(
                minified.get_dst_line(),
                minified.get_dst_col(),
                token.source_line,
                token.source_column,
                Some(source_id),
                name,
            );
            mapped_any = true;
        }
        if !mapped_any && readable_mappings.iter().any(|mapping| !mapping.tokens.is_empty()) {
            // Modules in this chunk DID carry real positions and none of them
            // survived composition — that is a bug in the composition, not an
            // honestly unmappable chunk, so it is refused rather than written out
            // as a map that quietly says nothing.
            return Err(format!(
                "source-map composition produced no honest mapping for minified chunk \
                 `{chunk_name}`: its modules carry real printer positions, but the \
                 minified->readable map resolved into none of them"
            ));
        }
        // A chunk whose modules are ALL bundler-synthesized (a CSS-module shim, an
        // asset stub, a virtual entry) has no original source to point at. Its map
        // is legitimately empty: every position in it came from code diffpack
        // wrote, and saying nothing is the truthful answer.
        Ok(builder.into_sourcemap().to_json_string())
    }

    /// Resolves one minified-map token against the chunk's readable tokens: the
    /// last readable token at or before it on the SAME readable line is the
    /// construct the minifier was printing. `None` means the position resolved
    /// into no module at all and must be written as an explicit unmapped marker.
    ///
    /// `names` means "the identifier this position had in the ORIGINAL source".
    /// The readable token's name is that by construction — Oxc records it from
    /// the source text under the node's span — so it wins. The minified map's
    /// name is only the identifier as it stood in the READABLE chunk. That is
    /// the original name whenever the lowering left it alone (the common case,
    /// and what makes a mangled stack trace readable), but where the lowering
    /// renamed it the readable name is a diffpack internal (`__import`,
    /// `__diffpack_import_7`) that never appeared in the user's file. So it is a
    /// fallback, not the first choice, and only accepted when the ORIGINAL
    /// source really has that identifier at the position being mapped
    /// (`source_lines` memoizes each module's line offsets for that check).
    fn resolve_minified_token<'a>(
        &'a self,
        minified: &oxc_sourcemap::Token,
        minified_map: &'a oxc_sourcemap::SourceMap,
        readable: &[(MapToken, DenseModuleId)],
        hint: &mut usize,
        source_lines: &mut HashMap<DenseModuleId, Vec<usize>>,
    ) -> Option<ResolvedMinifiedToken<'a>> {
        let position = (minified.get_src_line(), minified.get_src_col());
        // Minified tokens arrive in generated order and their readable positions
        // are nearly monotone, so the previous token's partition point is almost
        // always within a few entries of this one's: searching outward from it
        // replaces ~20 cache-missing probes of a full binary search per token
        // (millions of tokens for a large chunk) with 2-4 local ones. Exact by
        // construction — `partition_point_from_hint` returns precisely
        // `partition_point`'s answer for every hint.
        let candidate = partition_point_from_hint(readable, position, *hint);
        *hint = candidate;
        if candidate == 0 {
            return None;
        }
        let (token, dense) = &readable[candidate - 1];
        // A readable token only speaks for positions on its OWN line; past the
        // end of that line the minifier is printing something this chunk's
        // readable map says nothing about.
        if token.generated_line != minified.get_src_line() {
            return None;
        }
        let module = self.modules[*dense].as_ref()?;
        let map = module.map.as_ref()?;
        let name = match token
            .name
            .and_then(|index| map.names().get(index as usize))
            .filter(|name| is_identifier(name))
        {
            Some(name) => Some(name.as_str()),
            None => minified
                .get_name_id()
                .and_then(|index| minified_map.get_name(index))
                .filter(|candidate| {
                    let text = map.source_text(&module.source);
                    let lines = source_lines
                        .entry(*dense)
                        .or_insert_with(|| line_starts(text));
                    identifier_at(text, lines, token.source_line, token.source_column)
                        == Some(*candidate)
                }),
        };
        Some(ResolvedMinifiedToken {
            dense: *dense,
            source_line: token.source_line,
            source_column: token.source_column,
            name,
        })
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
                let entry = PathBuf::from(ResourceId::parse(self.ids[self.entry].as_ref()).path);
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
                for id in &self.ids {
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
        let resource = ResourceId::parse(self.ids[dense].as_ref());
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
                self.target.label(),
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
        let module_count = self.modules.len();
        let reachable: HashSet<DenseModuleId> = reachable
            .iter()
            .filter_map(|id| self.indices.get(id.as_str()).copied())
            .filter(|&index| self.modules[index].is_some())
            .collect();

        let mut live = vec![false; module_count];
        let mut used = vec![ExportDemand::default(); module_count];
        let mut queue: VecDeque<DenseModuleId> = VecDeque::new();

        fn mark_live(
            index: DenseModuleId,
            reachable: &HashSet<DenseModuleId>,
            live: &mut [bool],
            queue: &mut VecDeque<DenseModuleId>,
        ) {
            if reachable.contains(&index) && !live[index] {
                live[index] = true;
                queue.push_back(index);
            }
        }

        fn add_used(
            index: DenseModuleId,
            all: bool,
            names: &[String],
            reachable: &HashSet<DenseModuleId>,
            live: &mut [bool],
            used: &mut [ExportDemand],
            queue: &mut VecDeque<DenseModuleId>,
        ) {
            if !reachable.contains(&index) {
                return;
            }
            let mut changed = false;
            if all && !used[index].all {
                used[index].all = true;
                changed = true;
            }
            for name in names {
                changed |= used[index].names.insert(name.clone());
            }
            if changed {
                // A used export means the module's body must run to define it.
                live[index] = true;
                queue.push_back(index);
            }
        }

        // Seed: the entry always runs, and keeps whatever it re-exports (an app
        // entry is not a barrel, but this is the conservative, never-over-shake
        // choice).
        if reachable.contains(&self.entry) {
            live[self.entry] = true;
            used[self.entry].all = true;
            queue.push_back(self.entry);
        }

        while let Some(source) = queue.pop_front() {
            let Some(module) = self.modules[source].as_ref() else {
                continue;
            };
            let targets: HashMap<&str, DenseModuleId> = module
                .dependencies
                .iter()
                .map(|(specifier, target, _)| (specifier.as_str(), *target))
                .collect();

            // A dynamic import from live code roots its own chunk, which keeps
            // its full namespace.
            for (_, target, demand) in &module.dependencies {
                if demand.dynamic {
                    mark_live(*target, &reachable, &mut live, &mut queue);
                    add_used(*target, true, &[], &reachable, &mut live, &mut used, &mut queue);
                }
            }

            // Every static edge evaluates a module the flag does not authorize
            // dropping, so its side effects run (this covers bare side-effect
            // imports and re-exports of side-effectful modules alike).
            for (_, target, demand) in &module.dependencies {
                if !demand.deferred()
                    && self.modules[*target].as_ref().is_some_and(|state| !state.droppable)
                {
                    mark_live(*target, &reachable, &mut live, &mut queue);
                }
            }

            let liveness = &module.liveness;
            let empty_liveness = liveness.exports.is_empty()
                && liveness.reexports.is_empty()
                && liveness.star_reexports.is_empty()
                && liveness.body_uses.is_empty();
            if empty_liveness {
                // A synthesized module (route split, manifest, resolver) or any
                // module without captured export structure keeps every static
                // dependency it names — conservative, never over-shaking.
                for (specifier, target, demand) in &module.dependencies {
                    if !demand.deferred() {
                        let _ = specifier;
                        add_used(
                            *target,
                            demand.all,
                            &demand.names,
                            &reachable,
                            &mut live,
                            &mut used,
                            &mut queue,
                        );
                    }
                }
                continue;
            }

            // Body uses apply unconditionally now that the module is live.
            for body_use in &liveness.body_uses {
                if let Some(&target) = targets.get(body_use.specifier.as_str()) {
                    add_used(
                        target,
                        body_use.all,
                        &body_use.names,
                        &reachable,
                        &mut live,
                        &mut used,
                        &mut queue,
                    );
                }
            }

            // Snapshot this module's used exports; `add_used` on other modules
            // never shrinks it, and a self-update re-enqueues `source`.
            let source_all = used[source].all;
            let source_names = used[source].names.clone();

            // A re-export forwards demand to its source only when the forwarded
            // export is itself used.
            for reexport in &liveness.reexports {
                if (source_all || source_names.contains(&reexport.exported))
                    && let Some(&target) = targets.get(reexport.specifier.as_str())
                {
                    if reexport.imported == "*" {
                        add_used(target, true, &[], &reachable, &mut live, &mut used, &mut queue);
                    } else {
                        add_used(
                            target,
                            false,
                            std::slice::from_ref(&reexport.imported),
                            &reachable,
                            &mut live,
                            &mut used,
                            &mut queue,
                        );
                    }
                }
            }

            // A bare `export * from S` forwards a used name to S only when the
            // name is not one this module defines or explicitly re-exports (those
            // are already accounted for), i.e. it must have come from a star.
            for specifier in &liveness.star_reexports {
                let Some(&target) = targets.get(specifier.as_str()) else {
                    continue;
                };
                if source_all {
                    add_used(target, true, &[], &reachable, &mut live, &mut used, &mut queue);
                } else {
                    let names: Vec<String> = source_names
                        .iter()
                        .filter(|name| {
                            name.as_str() != "default" && !liveness.exports.contains(name)
                        })
                        .cloned()
                        .collect();
                    if !names.is_empty() {
                        add_used(target, false, &names, &reachable, &mut live, &mut used, &mut queue);
                    }
                }
            }
        }

        live.iter()
            .enumerate()
            .filter(|(_, is_live)| **is_live)
            .map(|(index, _)| self.ids[index].to_string())
            .collect()
    }

    fn export_demands(&self, sources: &[DenseModuleId]) -> Vec<ExportDemand> {
        let mut demands = vec![ExportDemand::default(); self.modules.len()];
        for &source in sources {
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

/// The native Node HTTP runtime emitted alongside the server module graph. Each
/// is a real `.mjs` template (authored under `src/server_runtime/`, embedded at
/// build time) written verbatim next to `server/server.mjs`:
///
/// - `index.mjs`         the `node:http` entry: PORT/HOST listener + wiring.
/// - `_ssr/node-adapter.mjs`  Node <-> Web `Request`/`Response` adapter + static
///   serving of the sibling `public/` assets.
/// - `_ssr/ssr.mjs`      resolves the app's SSR fetch handler from `server.mjs`.
/// - `_ssr/router.mjs`   re-exports the native TanStack Start route manifest.
///
/// The two `_ssr/*` filenames and `_ssr/router.mjs` also satisfy the acceptance
/// gates that require server artifacts whose names contain `ssr` and `router`.
const SERVER_RUNTIME_FILES: &[(&str, &str)] = &[
    (
        "index.mjs",
        include_str!("server_runtime/index.mjs"),
    ),
    (
        "_ssr/node-adapter.mjs",
        include_str!("server_runtime/_ssr/node-adapter.mjs"),
    ),
    (
        "_ssr/ssr.mjs",
        include_str!("server_runtime/_ssr/ssr.mjs"),
    ),
    (
        "_ssr/router.mjs",
        include_str!("server_runtime/_ssr/router.mjs"),
    ),
];

/// Writes the native server runtime entry files (see [`SERVER_RUNTIME_FILES`])
/// into an already-emitted `server_dir`. Called at emit time (off the
/// incremental hot path); each file is a static template that references its
/// siblings (`server.mjs`, `_tanstack-start-manifest_v.mjs`, `../public`) by
/// relative path, so no per-build interpolation is required.
fn write_server_runtime_entry(server_dir: &Path, hmr: bool) -> Result<Vec<PathBuf>, String> {
    let mut written = Vec::with_capacity(SERVER_RUNTIME_FILES.len());
    for (relative, contents) in SERVER_RUNTIME_FILES {
        let path = server_dir.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
        }
        // DEV-ONLY: the HMR SSR entry resolves the app fetch handler from a global
        // the control endpoint refreshes on each server edit (soft in-process
        // reload), so a server change is served without restarting Node. Production
        // uses the static template unchanged.
        let contents = if hmr && *relative == "_ssr/ssr.mjs" {
            HMR_SSR_ENTRY
        } else {
            contents
        };
        write_if_changed(&path, contents.as_bytes())?;
        written.push(path);
    }
    Ok(written)
}

/// DEV-ONLY replacement for `_ssr/ssr.mjs`: resolves the SSR fetch handler from
/// `globalThis.__diffpack_ssr_entry` (republished by the HMR control endpoint after
/// an in-process soft reload) so each request uses the latest app graph, falling
/// back to the boot handler before the first edit.
const HMR_SSR_ENTRY: &str = r#"import serverEntry from "../server.mjs";

export function resolveFetch(entry) {
  const seen = new Set();
  const queue = [entry];
  while (queue.length > 0) {
    const candidate = queue.shift();
    if (candidate == null || seen.has(candidate)) continue;
    seen.add(candidate);
    if (typeof candidate === "function") return candidate;
    if (typeof candidate.fetch === "function") return candidate.fetch.bind(candidate);
    if (typeof candidate === "object") queue.push(candidate.default);
  }
  throw new Error("diffpack ssr: ./server.mjs default export exposes no fetch handler");
}

export const fetch = (request) =>
  resolveFetch(globalThis.__diffpack_ssr_entry || serverEntry)(request);
export default { fetch };
"#;

/// Writes `bytes` to `path` only when the file's current contents differ, so an
/// unchanged output (a cache-reused chunk, an already-copied asset) is never
/// needlessly rewritten. Correctness is unchanged from an unconditional write:
/// the file always ends holding exactly `bytes`.
fn write_if_changed(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Ok(existing) = fs::read(path)
        && existing == bytes
    {
        return Ok(());
    }
    fs::write(path, bytes).map_err(|error| format!("cannot write {}: {error}", path.display()))
}

/// Deletes every file under `root` that is not in `keep`, then removes any
/// directory left empty. This replaces the old "wipe the whole output tree"
/// step: unchanged chunks stay on disk (already written by this emit), while
/// files that are no longer part of the build are removed, so no stale output
/// ever lingers.
/// Remove every file under `root` not in `keep`, then prune the now-empty
/// directories (never `root` itself). The public multi-page entry point: a
/// MULTI-PAGE build accumulates each page's written set (from
/// [`Bundler::emit_web_written`]) into one `keep` set and prunes once, so a stale
/// file from a prior build is deleted while no page clobbers another's output.
pub fn prune_web_output(root: &Path, keep: &BTreeSet<PathBuf>) -> Result<(), String> {
    prune_stale_files(root, keep)
}

fn prune_stale_files(root: &Path, keep: &BTreeSet<PathBuf>) -> Result<(), String> {
    if !root.exists() {
        return Ok(());
    }
    let mut directories = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(directory) = stack.pop() {
        directories.push(directory.clone());
        let entries = fs::read_dir(&directory)
            .map_err(|error| format!("cannot read {}: {error}", directory.display()))?;
        for entry in entries {
            let entry =
                entry.map_err(|error| format!("cannot read {}: {error}", directory.display()))?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if !keep.contains(&path) {
                fs::remove_file(&path)
                    .map_err(|error| format!("cannot remove {}: {error}", path.display()))?;
            }
        }
    }
    // Remove now-empty directories deepest-first (never the root itself).
    directories.sort_by_key(|directory| std::cmp::Reverse(directory.components().count()));
    for directory in directories {
        if directory == root {
            continue;
        }
        if fs::read_dir(&directory)
            .map(|mut entries| entries.next().is_none())
            .unwrap_or(false)
        {
            fs::remove_dir(&directory)
                .map_err(|error| format!("cannot remove {}: {error}", directory.display()))?;
        }
    }
    Ok(())
}

/// Folds an optional runtime id into a chunk render key, distinguishing `None`
/// from `Some(0)`.
fn hash_optional_id(hasher: &mut DefaultHasher, id: Option<usize>) {
    match id {
        Some(value) => {
            1u8.hash(hasher);
            value.hash(hasher);
        }
        None => 0u8.hash(hasher),
    }
}

/// Folds an aggregated export demand into a chunk render key. Names are sorted so
/// the key is order-independent (the demand is built from an unordered set).
fn hash_export_demand(hasher: &mut DefaultHasher, demand: &ExportDemand) {
    demand.all.hash(hasher);
    let mut names = demand.names.iter().collect::<Vec<_>>();
    names.sort();
    names.hash(hasher);
}

/// Which modules the registry runtime must render as `async` factories (indexed
/// by dense module id), and whether any does at all. `any == false` is the
/// overwhelmingly common case and makes every async-related runtime line, cache
/// key contribution and code rewrite a no-op, so a bundle with no top-level
/// `await` anywhere is byte-identical to what it was before async support.
#[derive(Default)]
struct AsyncModules {
    flags: Vec<bool>,
    any: bool,
}

impl AsyncModules {
    fn is_async(&self, dense: DenseModuleId) -> bool {
        self.any && self.flags.get(dense).copied().unwrap_or(false)
    }
}

/// How one module's lowered code reaches a given specifier, and therefore
/// whether that import site can be turned into an `await`. See
/// [`Bundler::async_module_closure`] for what each variant means for the build.
#[derive(Debug, Eq, PartialEq)]
enum AwaitableImport {
    /// A marked top-level import statement — awaitable.
    Statement,
    /// A top-level `__reExport(exports, require.esm(...))` — awaitable.
    ReExportAll,
    /// `export * as ns from ...`, lowered to a lazy getter — not awaitable.
    LazyNamespace,
    /// A CommonJS `require(...)` call, possibly nested in a function — not
    /// awaitable.
    BareRequire,
    /// The specifier is not statically required at all (a dynamic `import()`
    /// only, or a shaken-away site).
    None,
}

impl AwaitableImport {
    fn classify(code: &str, specifier: &str) -> Self {
        let quoted = quote(specifier);
        if code.contains(&format!("/*__diffpack_import:{quoted}__*/")) {
            return Self::Statement;
        }
        if code.contains(&format!("__reExport(exports,require.esm({quoted}));")) {
            return Self::ReExportAll;
        }
        if code.contains(&format!("()=>require.esm({quoted})")) {
            return Self::LazyNamespace;
        }
        if code.contains(&format!("require({quoted})")) {
            return Self::BareRequire;
        }
        Self::None
    }
}

/// Rewrites every top-level import site for `specifier` in an ASYNC module's
/// lowered code so it waits for the (async) target's initialisation to settle:
///
/// ```text
/// /*__diffpack_import:"./x"__*/ns=require.esm("./x");   ->  ...ns=await require.esmAsync("./x");
/// /*__diffpack_import:"./x"__*/require("./x");          ->  ...await require.async("./x");
/// __reExport(exports,require.esm("./x"));               ->  __reExport(exports,await require.esmAsync("./x"));
/// ```
///
/// The marked forms are rewritten anchored to their marker, so only the
/// statement the marker introduces is touched — never a `require("./x")` that
/// happens to sit inside a function body further down. No newline is added or
/// removed, so the module's generated-line span (and its source map) is
/// unchanged.
fn await_async_imports(code: &str, specifier: &str) -> String {
    let quoted = quote(specifier);
    let marker = format!("/*__diffpack_import:{quoted}__*/");
    let esm_call = format!("require.esm({quoted})");
    let plain_call = format!("require({quoted})");
    let mut out = String::with_capacity(code.len() + 32);
    let mut rest = code;
    while let Some(position) = rest.find(&marker) {
        let (head, tail) = rest.split_at(position + marker.len());
        out.push_str(head);
        let end = tail.find('\n').unwrap_or(tail.len());
        let (statement, after) = tail.split_at(end);
        if let Some(index) = statement.find(&esm_call) {
            out.push_str(&statement[..index]);
            out.push_str(&format!("await require.esmAsync({quoted})"));
            out.push_str(&statement[index + esm_call.len()..]);
        } else if let Some(index) = statement.find(&plain_call) {
            out.push_str(&statement[..index]);
            out.push_str(&format!("await require.async({quoted})"));
            out.push_str(&statement[index + plain_call.len()..]);
        } else {
            out.push_str(statement);
        }
        rest = after;
    }
    out.push_str(rest);
    out.replace(
        &format!("__reExport(exports,require.esm({quoted}));"),
        &format!("__reExport(exports,await require.esmAsync({quoted}));"),
    )
}

/// Statement-level tree shake of one module's lowered code, driven by the
/// cross-chunk export demand and TRANSITIVE local liveness: a removable
/// (obviously-pure, marker-wrapped) declaration is kept only when a demanded
/// export, an impure statement, or another LIVE declaration references one of
/// its names — a helper used solely by dead exports falls with them. Liveness
/// is a fixpoint over identifier occurrences in retained text; scanning raw
/// text is conservative in the safe direction (a name inside a string keeps
/// its declaration alive).
/// Additionally reports which line of `code` each
/// surviving line came from when `track_lines` is set. The shake only ever drops
/// whole lines or strips a bundler MARKER prefix off one, so the surviving lines
/// carry their map positions unchanged except for that prefix, which is recorded
/// as a column edit. Marker lines are pure bundler glue and never carry a token,
/// but the edit is recorded anyway so the accounting is complete rather than
/// merely believed.
fn shake_module_code(
    code: &str,
    demand: &ExportDemand,
    pruned_imports: &HashSet<String>,
    track_lines: bool,
) -> (String, Option<LineTrack>) {
    enum Segment<'a> {
        /// An unconditional line (impure statement, runtime call, ...).
        Keep(&'a str),
        /// A `/*__diffpack_import:spec__*/` line, dropped when the import was pruned.
        Import { line: &'a str },
        /// A `/*__diffpack_export:...*/` getter, kept only under demand.
        Export { statement: &'a str },
        /// A removable declaration block: its bound names and verbatim lines.
        Declaration { names: Vec<&'a str>, lines: Vec<&'a str> },
    }

    let mut segments: Vec<Segment> = Vec::new();
    // The line of `code` each segment line came from, and how many columns were
    // stripped off its front, parallel to the text the segment carries.
    let mut open_declaration: Option<(Vec<&str>, Vec<&str>)> = None;
    let mut origins: Vec<Vec<(usize, u32)>> = Vec::new();
    let mut open_origins: Vec<(usize, u32)> = Vec::new();
    for (line_index, line) in code.lines().enumerate() {
        if let Some((_, lines)) = open_declaration.as_mut() {
            if line == "/*__diffpack_decl_end__*/" {
                let (names, lines) = open_declaration.take().expect("declaration is open");
                segments.push(Segment::Declaration { names, lines });
                origins.push(std::mem::take(&mut open_origins));
            } else {
                lines.push(line);
                open_origins.push((line_index, 0));
            }
            continue;
        }
        if let Some(names) = line
            .strip_prefix("/*__diffpack_decl:")
            .and_then(|line| line.strip_suffix("__*/"))
        {
            open_declaration = Some((names.split(',').collect(), Vec::new()));
            continue;
        }
        if let Some(marked) = line.strip_prefix("/*__diffpack_import:")
            && let Some((specifier, import_code)) = marked.split_once("__*/")
            && let Ok(specifier) = serde_json::from_str::<String>(specifier)
        {
            if !pruned_imports.contains(&specifier) {
                let stripped = crate::source_map::utf16_len(
                    &line[..line.len() - import_code.len()],
                );
                segments.push(Segment::Import { line: import_code });
                origins.push(vec![(line_index, stripped)]);
            }
            continue;
        }
        if let Some(marked) = line.strip_prefix("/*__diffpack_export:")
            && let Some((name, statement)) = marked.split_once("__*/")
        {
            if demand.includes(name) {
                let stripped =
                    crate::source_map::utf16_len(&line[..line.len() - statement.len()]);
                segments.push(Segment::Export { statement });
                origins.push(vec![(line_index, stripped)]);
            }
            continue;
        }
        segments.push(Segment::Keep(line));
        origins.push(vec![(line_index, 0)]);
    }
    // An unterminated block would mean the transform emitted markers this parse
    // does not understand; keep its lines rather than silently dropping code.
    if let Some((_, lines)) = open_declaration.take() {
        for (line, origin) in lines.into_iter().zip(open_origins) {
            segments.push(Segment::Keep(line));
            origins.push(vec![origin]);
        }
    }

    // Map each removable name to its declaration segment.
    let mut owner_of: HashMap<&str, usize> = HashMap::new();
    for (index, segment) in segments.iter().enumerate() {
        if let Segment::Declaration { names, .. } = segment {
            for name in names {
                owner_of.insert(name, index);
            }
        }
    }

    // Seed liveness from everything that unconditionally executes, then follow
    // references to a fixpoint.
    let mut live = vec![false; segments.len()];
    let mut queue: Vec<usize> = Vec::new();
    let mark = |index: usize, live: &mut Vec<bool>, queue: &mut Vec<usize>| {
        if !live[index] {
            live[index] = true;
            queue.push(index);
        }
    };
    for (index, segment) in segments.iter().enumerate() {
        match segment {
            Segment::Keep(_) | Segment::Import { .. } | Segment::Export { .. } => {
                mark(index, &mut live, &mut queue);
            }
            Segment::Declaration { names, .. } => {
                // A demanded export name declared directly (the flat path's
                // `export{name}` footer references it outside any segment).
                if names.iter().any(|name| demand.includes(name)) {
                    mark(index, &mut live, &mut queue);
                }
            }
        }
    }
    while let Some(index) = queue.pop() {
        let scan = |text: &str, live: &mut Vec<bool>, queue: &mut Vec<usize>| {
            for word in identifier_runs(text) {
                if let Some(&owner) = owner_of.get(word)
                    && !live[owner]
                {
                    live[owner] = true;
                    queue.push(owner);
                }
            }
        };
        match &segments[index] {
            Segment::Keep(line) => scan(line, &mut live, &mut queue),
            Segment::Import { line } => scan(line, &mut live, &mut queue),
            Segment::Export { statement } => scan(statement, &mut live, &mut queue),
            Segment::Declaration { lines, .. } => {
                for line in lines {
                    scan(line, &mut live, &mut queue);
                }
            }
        }
    }

    let mut output = String::with_capacity(code.len());
    let mut track = track_lines.then(LineTrack::default);
    for (index, segment) in segments.iter().enumerate() {
        if !live[index] {
            continue;
        }
        if let Some(track) = track.as_mut() {
            for &(line, stripped) in &origins[index] {
                let mut origin = crate::source_map::LineOrigin {
                    source_line: Some(line as u32),
                    edits: Vec::new(),
                };
                if stripped > 0 {
                    origin.edits.push(crate::source_map::ColumnEdit {
                        column: 0,
                        removed: stripped,
                        inserted: 0,
                    });
                }
                track.push(origin);
            }
        }
        match segment {
            Segment::Keep(line) => {
                output.push_str(line);
                output.push('\n');
            }
            Segment::Import { line } => {
                output.push_str(line);
                output.push('\n');
            }
            Segment::Export { statement } => {
                output.push_str(statement);
                output.push('\n');
            }
            Segment::Declaration { lines, .. } => {
                for line in lines {
                    output.push_str(line);
                    output.push('\n');
                }
            }
        }
    }
    (output, track)
}

/// Iterator over the maximal identifier-character runs (`[A-Za-z0-9_$]+`) in
/// `text` that could be JavaScript identifiers (not starting with a digit).
fn identifier_runs(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '$'))
        .filter(|run| !run.is_empty() && !run.starts_with(|c: char| c.is_ascii_digit()))
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

/// True for a module diffpack itself generated into the project (`.diffpack-next/`,
/// `.diffpack-next-pages/`). Those files live INSIDE the app root, so the app's own
/// tsconfig `include` claims them, but they are diffpack's source written against
/// React — handing them the app's `jsxImportSource` would lower diffpack's own
/// runtime against a package it never imports.
fn is_generated_adapter_module(path: &Path) -> bool {
    path.components().any(|component| {
        matches!(
            component.as_os_str().to_str(),
            Some(crate::next_adapter::ADAPTER_DIR | crate::next_pages::ADAPTER_DIR)
        )
    })
}

/// How `path`'s JSX must be lowered: the `jsx` / `jsxImportSource` / `jsxFactory` /
/// `jsxFragmentFactory` of the tsconfig/jsconfig that CONFIGURES the file, with the
/// build's own settings (`overrides`, from `vite.config` and its plugins) winning
/// field by field — Vite's exact precedence.
///
/// Applicability, not type-checking: see [`crate::jsx_project_config`] for why the
/// resolver's own `find_tsconfig` is the wrong question here (it excludes `.jsx`
/// from a config without `allowJs`, splitting one app across two JSX runtimes, and
/// never reads `jsconfig.json` at all). Files under `node_modules` get no config,
/// which is what keeps a dependency's `.tsx` off the app's import source.
/// The compilation contract one file inherits from the tsconfig/jsconfig that owns
/// it: how its JSX is lowered and how its `@decorator`s are. Read together because
/// they come from ONE config and finding that config walks the directory tree —
/// asking twice would walk it twice per module.
fn project_config_for(
    resolver: &Resolver,
    path: &Path,
    overrides: &crate::transform::JsxConfig,
) -> Result<crate::transform::ProjectConfig, String> {
    if is_generated_adapter_module(path) {
        return Ok(crate::transform::ProjectConfig {
            jsx: crate::transform::JsxConfig::default().overridden_by(overrides),
            decorators: crate::transform::DecoratorConfig::default(),
        });
    }
    let tsconfig = crate::jsx_project_config::owning_config(resolver, path)?;
    let mut resolved = crate::transform::JsxConfig::default();
    let mut decorators = crate::transform::DecoratorConfig::default();
    if let Some(tsconfig) = tsconfig {
        let compiler_options = &tsconfig.compiler_options;
        // TypeScript's own defaults for a config that omits them: decorators are
        // Stage 3, no metadata, and `strictNullChecks` is off unless it (or the
        // `strict` family switch that implies it) is on.
        decorators = crate::transform::DecoratorConfig {
            legacy: compiler_options.experimental_decorators.unwrap_or(false),
            emit_metadata: compiler_options.emit_decorator_metadata.unwrap_or(false),
            strict_null_checks: compiler_options
                .strict_null_checks
                .or(compiler_options.strict)
                .unwrap_or(false),
        };
        resolved.runtime = match compiler_options.jsx.as_deref() {
            None => None,
            Some("react-jsx" | "react-jsxdev") => Some(crate::transform::JsxRuntime::Automatic),
            Some("react") => Some(crate::transform::JsxRuntime::Classic),
            // `preserve`/`react-native` tell TYPESCRIPT to emit JSX unchanged and
            // leave the lowering to a downstream tool — which is this bundler. A
            // browser cannot run JSX, so there is nothing to preserve; the automatic
            // runtime is what every such toolchain (Next's SWC loader on
            // create-next-app's `"jsx": "preserve"`, Vite's oxc pass) actually emits.
            Some("preserve" | "react-native") => Some(crate::transform::JsxRuntime::Automatic),
            Some(other) => {
                return Err(format!(
                    "{}: unsupported \"jsx\" value {other:?} (expected one of \"react-jsx\", \
                     \"react-jsxdev\", \"react\", \"preserve\", \"react-native\"), which owns {}",
                    tsconfig.path.display(),
                    path.display(),
                ));
            }
        };
        resolved.import_source.clone_from(&compiler_options.jsx_import_source);
        resolved.factory.clone_from(&compiler_options.jsx_factory);
        resolved.fragment_factory.clone_from(&compiler_options.jsx_fragment_factory);
    }
    Ok(crate::transform::ProjectConfig {
        jsx: resolved.overridden_by(overrides),
        decorators,
    })
}

fn load_uncached(
    resolver: &Resolvers,
    resolution_cache: &ResolutionCache,
    path: &Path,
    target: Target,
    hmr: bool,
    source_maps: bool,
) -> Result<LoadedModule, String> {
    let id = path.to_string_lossy();
    // A build-generated virtual module (its source is not on disk) claims this id
    // first.
    if let Some(source) = resolution_cache.virtual_module_source(&id) {
        let special = synthesize_virtual_module(source)?;
        let mut diagnostics = Vec::new();
        let resolved = resolve_special_dependencies(
            resolver,
            resolution_cache,
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
            css: special.css,
            css_source_files: special.css_source_files,
            css_external_imports: special.css_external_imports,
            externals: resolved.externals,
            droppable: false,
            liveness: ModuleLiveness::default(),
            uses_top_level_await: false,
            uses_import_meta: false,
            uses_cjs_globals: false,
            uses_dirname: false,
            workers: Vec::new(),
            map: None,
        });
    }
    // A loader (query, stylesheet, or asset) may claim this id before it is ever
    // read as JavaScript.
    if let Some(special) = load_special_module(&id, path, target, resolution_cache) {
        let mut special = special?;
        // DEV-ONLY: instrument a `?tsr-split=<component>` route component with the
        // Fast Refresh footer (see [`crate::hmr`]). Never runs for `build-app`.
        if hmr
            && target == Target::Client
            && crate::hmr::is_refresh_boundary(
                Path::new(id.as_ref()),
                &[],
                "",
                resolution_cache.jsx_extensions,
            )
        {
            special
                .code
                .push_str(&crate::hmr::fast_refresh_footer(id.as_ref()));
        }
        let mut diagnostics = Vec::new();
        let resolved = resolve_special_dependencies(
            resolver,
            resolution_cache,
            &id,
            target,
            &special,
            &mut diagnostics,
        );
        // A `?worker` module registers its referenced entry as a worker chunk;
        // the key matches the `__diffpack_worker__<key>__` placeholder in the
        // synthesized constructor so the emit-step substitution resolves it.
        let resource = ResourceId::parse(&id);
        let workers = if resource.loader_kind() == Some(LoaderKind::Worker) {
            vec![(worker_import_key(&resource.path), PathBuf::from(&resource.path))]
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
            css: special.css,
            css_source_files: special.css_source_files,
            css_external_imports: special.css_external_imports,
            externals: resolved.externals,
            droppable: false,
            liveness: ModuleLiveness::default(),
            uses_top_level_await: false,
            uses_import_meta: false,
            uses_cjs_globals: false,
            uses_dirname: false,
            workers,
            map: None,
        });
    }
    let read_started = frontend_profile::start();
    let source = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    frontend_profile::finish(Phase::Read, read_started);
    let hash = content_hash(source.as_bytes());
    // A `.vue`/`.svelte` component is compiled to JavaScript by the app's own
    // compiler before anything below reads it as a module (see
    // [`precompile_component`]).
    let (component_code, language, component) =
        match precompile_component(path, &source, resolution_cache)? {
            Some(compiled) => (
                Some(compiled.code),
                compiled.language,
                compiled.side_effects,
            ),
            None => (
                None,
                crate::transform::SourceLanguage::FromPath,
                ComponentSideEffects::default(),
            ),
        };
    let module_text = component_code.as_deref().unwrap_or(source.as_str());
    let source = resolution_cache.apply_vite_replacements(path, module_text, target)?;
    let project_config = project_config_for(resolver, path, &resolution_cache.jsx)?;
    let mut transformed = crate::transform::transform_module_in_language(
        path,
        &source,
        target,
        hmr && target == Target::Client,
        resolution_cache.jsx_extensions,
        &project_config,
        language,
        source_maps,
    );
    // See the `&self` twin: a rewritten source is labelled as one.
    mark_rewritten_source(
        &mut transformed,
        component_code.is_some(),
        matches!(source, Cow::Owned(_)),
    );
    // DEV-ONLY Fast Refresh / `import.meta.hot` instrumentation (client only).
    if hmr {
        let before_refresh = std::mem::take(&mut transformed.code);
        let hot_rewritten = before_refresh.contains("import.meta.hot");
        transformed.code = crate::hmr::rewrite_import_meta_hot(&before_refresh, target);
        let mut preamble_lines = 0_u32;
        if target == Target::Client {
            let module_key = path.to_string_lossy();
            // See the `&self` twin above: the module-scoped `$RefreshReg$` goes on
            // EVERY instrumented client module, not only the accept boundaries.
            if crate::hmr::needs_fast_refresh_preamble(&transformed.code) {
                let preamble = crate::hmr::fast_refresh_preamble(&module_key);
                preamble_lines = preamble.bytes().filter(|byte| *byte == b'\n').count() as u32;
                transformed.code.insert_str(0, &preamble);
            }
            if crate::hmr::is_refresh_boundary(
                path,
                &transformed.liveness.exports,
                &source,
                resolution_cache.jsx_extensions,
            ) {
                transformed
                    .code
                    .push_str(&crate::hmr::fast_refresh_footer(&module_key));
            }
        }
        rebase_map_for_refresh(
            &mut transformed.map,
            &before_refresh,
            preamble_lines,
            hot_rewritten,
        );
    }
    let code_hash = content_hash(transformed.code.as_bytes());
    let mut diagnostics = source_diagnostics(path, &transformed.diagnostics);
    let (dependency_specifiers, dependency_demands) =
        component_dependencies(&transformed, &component);
    let dependencies = resolve_dependencies(
        resolver,
        resolution_cache,
        path,
        target,
        &dependency_specifiers,
        &dependency_demands,
        &mut diagnostics,
    );
    let droppable = module_droppable(path, &mut diagnostics);
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
        css: component.css,
        css_source_files: component.css_source_files,
        css_external_imports: component.css_external_imports,
        externals: dependencies.externals,
        droppable,
        liveness: transformed.liveness,
        uses_top_level_await: transformed.uses_top_level_await,
        uses_import_meta: transformed.uses_import_meta,
        uses_cjs_globals: transformed.uses_cjs_globals,
        uses_dirname: transformed.uses_dirname,
        workers: resolve_worker_entries(resolver, path, &transformed.workers)?,
        map: transformed.map,
    })
}

/// Keeps a module's map honest about WHICH TEXT its positions were measured
/// against. The transform is handed `source`, which is the file on disk only
/// when neither a component compiler (`.vue`/`.svelte`) nor Vite's
/// `import.meta`/`define`/dead-branch replacements rewrote it first. When one
/// did, the map is labelled as generated, so its `sources` entry names the
/// rewrite and its inlined content is the text the positions really index —
/// never a silent claim on the file.
fn mark_rewritten_source(
    transformed: &mut crate::transform::TransformResult,
    precompiled: bool,
    vite_rewritten: bool,
) {
    let Some(map) = transformed.map.as_mut() else {
        return;
    };
    if precompiled {
        map.mark_generated("component");
    } else if vite_rewritten {
        map.mark_generated("vite-replace");
    }
}

/// Re-bases a module's map across the DEV-ONLY Fast Refresh instrumentation,
/// which edits the lowered code after the printer produced its map.
///
/// `import.meta.hot` -> `module.hot` is an in-place rewrite (recorded as a column
/// edit, so a token inside it is dropped and one after it moves by the exact
/// amount the line shrank); the per-module preamble is one whole line inserted at
/// the top (every following line shifts by one); the self-accept footer starts
/// with a newline and is appended, so it changes no existing line at all.
fn rebase_map_for_refresh(
    map: &mut Option<ModuleSourceMap>,
    before: &str,
    preamble_lines: u32,
    hot_rewritten: bool,
) {
    let Some(current) = map.as_mut() else {
        return;
    };
    let lines = before.lines().count();
    let mut track = LineTrack::synthetic(preamble_lines as usize);
    track.extend(LineTrack::identity(lines));
    if hot_rewritten {
        // Recomputed rather than threaded through, so the columns come from the
        // same text the map describes.
        let mut edits = LineTrack::identity(lines);
        let _ = crate::source_map::replace_tracked(before, "import.meta.hot", "module.hot", &mut edits);
        let mut combined = LineTrack::synthetic(preamble_lines as usize);
        for index in 0..lines {
            combined.push(edits.line(index).cloned().expect("line in range"));
        }
        track = combined;
    }
    current.rebase(&track, preamble_lines as usize + lines);
}

/// Attributes a module's oxc diagnostics to its path, keeping each one's
/// severity so an error fails the build and a warning only prints.
fn source_diagnostics(
    path: &Path,
    diagnostics: &[crate::transform::TransformDiagnostic],
) -> Vec<Diagnostic> {
    diagnostics
        .iter()
        .map(|diagnostic| Diagnostic {
            kind: DiagnosticKind::Source {
                fatal: diagnostic.fatal,
            },
            message: format!("{}: {}", path.display(), diagnostic.message),
        })
        .collect()
}

/// Whether the module at `path` may be dropped when unused, per its nearest
/// `package.json`'s `sideEffects` field. An unsupported `sideEffects` glob is a
/// hard, specific error, surfaced as a build diagnostic; the module is then kept
/// (treated as non-droppable), never silently mis-classified.
fn module_droppable(path: &Path, diagnostics: &mut Vec<Diagnostic>) -> bool {
    match crate::side_effects::is_droppable(path) {
        Ok(droppable) => droppable,
        Err(error) => {
            diagnostics.push(Diagnostic {
                kind: DiagnosticKind::SideEffectsGlob,
                message: format!("{}: {error}", path.display()),
            });
            false
        }
    }
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
    /// See [`ModuleState::css_source_files`].
    css_source_files: Vec<PathBuf>,
    /// See [`ModuleState::css_external_imports`].
    css_external_imports: Vec<String>,
    /// Import specifiers and per-specifier demand the synthesized code carries.
    /// Empty for a leaf synthetic module (an asset URL, a `?raw` string, an
    /// extracted stylesheet). A route-split (`?tsr-split`) module, by contrast, is
    /// real JavaScript with `import`s (React, the route's own module-level deps),
    /// so its dependencies MUST become graph edges: otherwise its lowered
    /// `require(...)` calls have no runtime map entry and fall through to
    /// `requireNative`. That is invisible on the server (Node's `createRequire`
    /// resolves them from `node_modules`) but fatal in the browser, which has no
    /// `node_modules`. These are resolved by the load paths relative to the real
    /// source file (the route file), exactly like a normal module's imports.
    dependency_specifiers: Vec<String>,
    dependency_demands: Vec<DependencyDemand>,
}

/// Loads a non-JavaScript module when a loader applies to `path`/`id`: a query
/// loader (`?url`, `?raw`), a global stylesheet (`.css`), or a default asset
/// import (image/font/SVG/...). Returns `None` for an ordinary JS/TS module,
/// which the normal read-and-transform path then handles.
fn load_special_module(
    id: &str,
    path: &Path,
    target: Target,
    cache: &ResolutionCache,
) -> Option<Result<SpecialModule, String>> {
    let resource = ResourceId::parse(id);
    if resource.query.is_some() {
        return Some(synthesize_query_module_impl(
            &resource,
            target,
            &cache.base,
            cache.asset_inline_limit,
            cache.css_preprocess.root_path(),
        ));
    }
    let postcss = cache.css_preprocess.postcss.as_deref();
    if crate::css::is_css_module_path(path) {
        return Some(load_css_module(path, target, postcss));
    }
    if is_css_path(path) {
        return Some(load_stylesheet(path, postcss));
    }
    if crate::sass::is_scss_path(path) {
        return Some(load_scss(path, target, &cache.scss, postcss));
    }
    if crate::less_stylus::is_less_or_stylus_path(path) {
        return Some(load_less_or_stylus(
            path,
            target,
            cache.css_preprocess.root_path(),
            postcss,
        ));
    }
    if is_asset_path(path) {
        return Some(synthesize_asset_url(
            path.to_path_buf(),
            &cache.base,
            cache.asset_inline_limit,
            cache.image_import_shape,
        ));
    }
    // Nothing above claimed it. Falling through to `None` would mean "read it as
    // JavaScript", which for an `.astro`/`.graphql` file produces a parse error
    // in the app's own source instead of naming diffpack's gap. (`.vue` and
    // `.svelte` ARE claimed — by the JS load path, which compiles them first;
    // see [`precompile_component`].)
    if let Some(unhandled) = unhandled_source(path) {
        return Some(Err(unhandled_source_message(path, &unhandled)));
    }
    None
}

/// A `.vue`/`.svelte` component after its own framework compiler has run: the
/// JavaScript to parse in place of the file's text, plus everything its
/// `<style>` blocks contributed once they went through the ordinary CSS
/// pipeline (rebased `url(...)`s, PostCSS, `@import` edges).
struct PrecompiledComponent {
    /// The compiled JavaScript, still carrying the component's own imports.
    code: String,
    /// How [`Self::code`] must be parsed — a Vue SFC with `<script lang="ts">`
    /// compiles to TypeScript.
    language: crate::transform::SourceLanguage,
    /// Everything the component contributes BESIDES its JavaScript.
    side_effects: ComponentSideEffects,
}

/// What a component's `<style>` blocks contribute to its module, once they have
/// been through the ordinary CSS pipeline. A module that is not a component
/// contributes nothing, which is exactly [`Default`].
#[derive(Default)]
struct ComponentSideEffects {
    css: Option<String>,
    css_source_files: Vec<PathBuf>,
    css_external_imports: Vec<String>,
    assets: Vec<AssetEmit>,
    /// `@import` targets the component's styles pull in. They are appended to
    /// the module's JavaScript dependencies so the imported stylesheet is a real
    /// graph edge — the same treatment a `.css` module's `@import` gets.
    dependency_specifiers: Vec<String>,
    dependency_demands: Vec<DependencyDemand>,
}

/// The dependency specifiers and demands a module contributes: its JavaScript
/// imports, plus a component's style `@import`s. Borrows the transform's own
/// vectors for every module that is not a component — the common case must not
/// pay a copy for a feature it does not use.
fn component_dependencies<'a>(
    transformed: &'a crate::transform::TransformResult,
    component: &ComponentSideEffects,
) -> (Cow<'a, [String]>, Cow<'a, [DependencyDemand]>) {
    if component.dependency_specifiers.is_empty() {
        return (
            Cow::Borrowed(&transformed.dependencies),
            Cow::Borrowed(&transformed.dependency_demands),
        );
    }
    let mut specifiers = transformed.dependencies.clone();
    specifiers.extend(component.dependency_specifiers.iter().cloned());
    let mut demands = transformed.dependency_demands.clone();
    demands.extend(component.dependency_demands.iter().cloned());
    (Cow::Owned(specifiers), Cow::Owned(demands))
}

/// Compiles `path` when it is a single-file component, returning `None` for
/// every ordinary JavaScript/TypeScript module (the overwhelming majority — the
/// check is one extension comparison). The component's compiler is the app's
/// own; see [`crate::sfc`]. A failure to compile is a hard error naming the
/// file, never a fall-through to "parse it as JavaScript".
fn precompile_component(
    path: &Path,
    source: &str,
    cache: &ResolutionCache,
) -> Result<Option<PrecompiledComponent>, String> {
    let Some(framework) = crate::sfc::framework_for(path) else {
        return Ok(None);
    };
    let compiled = crate::sfc::compile(framework, path, source, cache.css_preprocess.root_path())?;
    let language = match compiled.language {
        crate::sfc::OutputLanguage::TypeScript => crate::transform::SourceLanguage::TypeScript,
        crate::sfc::OutputLanguage::JavaScript => crate::transform::SourceLanguage::JavaScript,
    };
    // The component's styles take the SAME path a hand-written stylesheet takes,
    // so an SFC `url(../a.png)` is rebased and hashed exactly like one in a
    // `.css` file. The JS stub that loader synthesizes is discarded: this
    // module's JavaScript is the component's own.
    let styles = match &compiled.css {
        Some(css) => Some(load_stylesheet_from_text(
            path,
            css,
            Vec::new(),
            cache.css_preprocess.postcss.as_deref(),
        )?),
        None => None,
    };
    let side_effects = match styles {
        Some(styles) => ComponentSideEffects {
            css: styles.css,
            css_source_files: styles.css_source_files,
            css_external_imports: styles.css_external_imports,
            assets: styles.assets,
            dependency_specifiers: styles.dependency_specifiers,
            dependency_demands: styles.dependency_demands,
        },
        None => ComponentSideEffects::default(),
    };
    Ok(Some(PrecompiledComponent {
        code: compiled.code,
        language,
        side_effects,
    }))
}

/// A Sass source (`.scss`): compiled natively to plain CSS first, then handed
/// to the SAME pipeline a hand-written CSS file takes — `.module.scss` through
/// the CSS Modules scoper, everything else through the global-stylesheet
/// loader. Every `@use`/`@import`ed partial (and the `additionalData` theme)
/// is recorded in `css_source_files`, so editing one re-derives this module.
fn load_scss(
    path: &Path,
    target: Target,
    options: &crate::sass::ScssOptions,
    postcss: Option<&crate::postcss::Postcss>,
) -> Result<SpecialModule, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let compiled = crate::sass::compile_scss(path, &text, options)?;
    if crate::sass::is_scss_module_path(path) {
        load_css_module_from_text(path, &compiled.css, compiled.loaded_files, target, postcss)
    } else {
        load_stylesheet_from_text(path, &compiled.css, compiled.loaded_files, postcss)
    }
}

/// A Less/Stylus source: compiled to plain CSS by the app's own preprocessor
/// (`node`, cwd = project root), then handed to the SAME pipeline a hand-written
/// CSS file takes — `.module.less`/`.module.styl` through the CSS Modules scoper,
/// everything else through the global-stylesheet loader. Every `@import`ed file
/// the preprocessor pulled in is recorded so editing it re-derives this module.
fn load_less_or_stylus(
    path: &Path,
    target: Target,
    root: Option<&Path>,
    postcss: Option<&crate::postcss::Postcss>,
) -> Result<SpecialModule, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let compiled = if crate::less_stylus::is_less_path(path) {
        crate::less_stylus::compile_less(path, &text, root)?
    } else {
        crate::less_stylus::compile_stylus(path, &text, root)?
    };
    if crate::less_stylus::is_css_module_path(path) {
        load_css_module_from_text(path, &compiled.css, compiled.loaded_files, target, postcss)
    } else {
        load_stylesheet_from_text(path, &compiled.css, compiled.loaded_files, postcss)
    }
}

/// Builds the module for a query-bearing id. `?url` emits a content-hashed asset
/// and exports its URL; `?raw` inlines the file contents as a string.
/// Recognized-but-unimplemented loaders (`?tsr-split`) and unrecognized queries
/// produce a specific, actionable error rather than a misleading filesystem read
/// failure.
fn synthesize_query_module_impl(
    resource: &ResourceId,
    target: Target,
    base: &str,
    asset_inline_limit: usize,
    root: Option<&Path>,
) -> Result<SpecialModule, String> {
    match resource.loader_kind() {
        // An explicit `?url` import is always the bare URL string (Vite semantics),
        // regardless of the build's default image-import shape.
        Some(LoaderKind::Url) => {
            synthesize_asset_url(PathBuf::from(&resource.path), base, asset_inline_limit, ImageImportShape::Url)
        }
        Some(LoaderKind::Raw) => synthesize_raw(Path::new(&resource.path)),
        Some(LoaderKind::TsrSplit) => synthesize_tsr_split(resource, target),
        Some(LoaderKind::CssMedia) => synthesize_css_media(resource),
        Some(LoaderKind::Worker) => synthesize_worker(resource),
        Some(LoaderKind::Inline) => synthesize_inline(Path::new(&resource.path)),
        Some(LoaderKind::WasmInit) => synthesize_wasm_init(resource, base, asset_inline_limit),
        Some(LoaderKind::PublicUrl) => synthesize_public_url(resource, base, root),
        None => Err(resource.unimplemented_loader_error()),
    }
}

/// A file under the project's `public/` directory, reached by a root-absolute
/// import (`import icons from "/icons.svg"`). The public directory is copied to
/// the site root VERBATIM, so the file is not read, hashed, or emitted here: the
/// module is exactly its public URL, `<base><path under public/>` — which is
/// what Vite's own root-absolute import yields.
///
/// The URL is derived from the resolved path, so a resolved id that is not under
/// `<root>/public` is a contradiction between the resolver and this loader and
/// fails loudly rather than minting a URL that resolves to nothing.
fn synthesize_public_url(
    resource: &ResourceId,
    base: &str,
    root: Option<&Path>,
) -> Result<SpecialModule, String> {
    let path = Path::new(&resource.path);
    let root = root.ok_or_else(|| {
        format!(
            "{}: a `?public-url` module needs the project root to derive its URL, and this \
             build has none",
            path.display()
        )
    })?;
    let public_dir = root.join("public");
    let relative = path.strip_prefix(&public_dir).map_err(|_| {
        format!(
            "{}: a `?public-url` module must live under {}",
            path.display(),
            public_dir.display()
        )
    })?;
    // Public URLs are `/`-separated regardless of the host filesystem.
    let url_path = relative
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/");
    let url = format!("{base}{url_path}");
    let synthetic = format!("export default {};\n", quote(&url));
    let transformed = transform_module(Path::new("diffpack-public-url.js"), &synthetic, Target::Server);
    Ok(SpecialModule {
        hash: content_hash(url.as_bytes()),
        code: transformed.code,
        flat_module: transformed.flat_module,
        assets: Vec::new(),
        css: None,
        css_source_files: Vec::new(),
        css_external_imports: Vec::new(),
        dependency_specifiers: Vec::new(),
        dependency_demands: Vec::new(),
    })
}

/// The deterministic worker key for a `?worker` import site, derived from the
/// resolved entry path. Defined once and used both to mint the
/// `__diffpack_worker__<key>__` placeholder inside the synthesized constructor
/// module ([`synthesize_worker`]) and to register the worker entry for bundling
/// (in [`ModuleGraph::load_module`]), so the two agree by construction.
fn worker_import_key(entry_path: &str) -> String {
    crate::transform::worker_key(Path::new(entry_path), "?worker")
}

/// A `?worker` module: `import W from './w.js?worker'` yields a default-exported
/// `Worker` constructor. The referenced entry is bundled as its own
/// self-contained browser chunk (registered as a worker entry in
/// [`ModuleGraph::load_module`], emitted under `assets/`), and the constructor
/// spawns a module worker at the emitted URL. The URL is a
/// `__diffpack_worker__<key>__` placeholder that the emit step substitutes with
/// the real public path, exactly like the `new Worker(new URL(...))` form.
///
/// `?worker&inline` (a blob-inlined worker) is refused with a specific error
/// rather than silently emitting a separate file the app did not ask for.
fn synthesize_worker(resource: &ResourceId) -> Result<SpecialModule, String> {
    if resource.query_has_flag("inline") {
        return Err(format!(
            "loader `?worker&inline` (blob-inlined workers) is not yet implemented \
             (requested for {}); use `?worker` for a separately emitted worker chunk",
            resource.path
        ));
    }
    let key = worker_import_key(&resource.path);
    // `new WorkerWrapper()` returns the constructed Worker: a constructor that
    // returns an object yields that object. `options` forwards a caller-supplied
    // `name`/`credentials` while keeping the module `type`, matching Vite's
    // `?worker` default-export shape.
    let placeholder = format!("__diffpack_worker__{key}__");
    let synthetic = format!(
        "export default function WorkerWrapper(options) {{\n  \
           return new Worker({}, {{ type: \"module\", ...options }});\n}}\n",
        quote(&placeholder),
    );
    let transformed = transform_module(Path::new("diffpack-worker.js"), &synthetic, Target::Client);
    Ok(SpecialModule {
        hash: content_hash(transformed.code.as_bytes()),
        code: transformed.code,
        flat_module: transformed.flat_module,
        assets: Vec::new(),
        css: None,
        css_source_files: Vec::new(),
        css_external_imports: Vec::new(),
        dependency_specifiers: Vec::new(),
        dependency_demands: Vec::new(),
    })
}

/// A `?inline` module: the asset is ALWAYS embedded as a `data:` URI (default
/// string export), regardless of the build's inline-size threshold. SVG keeps
/// its native `data:image/svg+xml` encoding; everything else is base64 under
/// its content type.
fn synthesize_inline(source_path: &Path) -> Result<SpecialModule, String> {
    let bytes = fs::read(source_path)
        .map_err(|error| format!("cannot read asset {}: {error}", source_path.display()))?;
    let data_uri = svg_data_url(source_path, &bytes).unwrap_or_else(|| {
        format!(
            "data:{};base64,{}",
            asset_mime_type(source_path),
            base64_encode(&bytes)
        )
    });
    let synthetic = format!("export default {};\n", quote(&data_uri));
    let transformed =
        transform_module(Path::new("diffpack-inline-asset.js"), &synthetic, Target::Server);
    Ok(SpecialModule {
        hash: content_hash(transformed.code.as_bytes()),
        code: transformed.code,
        flat_module: transformed.flat_module,
        assets: Vec::new(),
        css: None,
        css_source_files: Vec::new(),
        css_external_imports: Vec::new(),
        dependency_specifiers: Vec::new(),
        dependency_demands: Vec::new(),
    })
}

/// A `.wasm?init` module: `import init from './m.wasm?init'` yields an async
/// initializer `(imports) => Promise<WebAssembly.Instance>`. The `.wasm` byte
/// payload takes the SAME content-hashed `assets/` pipeline (and inline-limit
/// `data:` fast path) as a `?url` asset; the emitted/inlined URL is closed over
/// by the shared instantiation helper ([`wasm_helper.js`], ported from Vite).
fn synthesize_wasm_init(
    resource: &ResourceId,
    base: &str,
    inline_limit: usize,
) -> Result<SpecialModule, String> {
    let source_path = PathBuf::from(&resource.path);
    if source_path.extension().and_then(|value| value.to_str()) != Some("wasm") {
        return Err(format!(
            "loader `?init` applies only to `.wasm` files (requested for {})",
            resource.path
        ));
    }
    let bytes = fs::read(&source_path)
        .map_err(|error| format!("cannot read {}: {error}", source_path.display()))?;
    // Small modules inline as a `data:` URI (the helper base64-decodes and
    // instantiates directly); larger ones emit a content-hashed `assets/` file
    // served as `application/wasm`.
    let (url, assets) = if inline_limit > 0 && bytes.len() <= inline_limit {
        let data_uri = format!(
            "data:application/wasm;base64,{}",
            base64_encode(&bytes)
        );
        (data_uri, Vec::new())
    } else {
        let public_name = asset_public_name(&source_path, content_hash(&bytes));
        let url = format!("{base}assets/{public_name}");
        let emit = AssetEmit {
            source: source_path,
            public_name,
            tailwind_source: None,
            image_variants: None,
        };
        (url, vec![emit])
    };
    let helper = include_str!("wasm_helper.js");
    let synthetic = format!(
        "{helper}\nconst __diffpackWasmUrl = {};\n\
         export default (imports = {{}}) => __diffpackWasmInit(imports, __diffpackWasmUrl);\n",
        quote(&url),
    );
    let transformed = transform_module(Path::new("diffpack-wasm-init.js"), &synthetic, Target::Server);
    Ok(SpecialModule {
        hash: content_hash(transformed.code.as_bytes()),
        code: transformed.code,
        flat_module: transformed.flat_module,
        assets,
        css: None,
        css_source_files: Vec::new(),
        css_external_imports: Vec::new(),
        dependency_specifiers: Vec::new(),
        dependency_demands: Vec::new(),
    })
}

/// A `?media=<query>` module: a CSS file imported under a media query
/// (`@import './x.css' screen;`). Its emitted stylesheet text is the file's
/// content — imports inlined recursively, `url(...)`s resolved relative to each
/// contributing file, CSS Modules scoped — wrapped in `@media <query> { ... }`.
/// The module id carries the query, so `(file, media)` pairs dedup naturally
/// and the physical file's edits re-derive it via the derived-sibling scan.
fn synthesize_css_media(resource: &ResourceId) -> Result<SpecialModule, String> {
    let path = Path::new(&resource.path);
    if !is_css_path(path) {
        return Err(format!(
            "loader `?media` applies only to CSS files (requested for {})",
            resource.path
        ));
    }
    let media = resource
        .query
        .as_deref()
        .and_then(|query| query.strip_prefix("media="))
        .filter(|media| !media.trim().is_empty())
        .ok_or_else(|| {
            format!(
                "loader `?media` requires a media query value (requested for {})",
                resource.path
            )
        })?;
    let text = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let processed = crate::css::process_media_import(path, &text, media)?;
    Ok(SpecialModule {
        hash: content_hash(processed.css.as_bytes()),
        code: String::new(),
        flat_module: None,
        assets: css_assets_to_emits(processed.assets),
        css: Some(processed.css),
        css_source_files: processed.inlined_files,
        css_external_imports: Vec::new(),
        dependency_specifiers: Vec::new(),
        dependency_demands: Vec::new(),
    })
}

/// Converts CSS-referenced assets into the shared emit records; they join the
/// same content-hashed `assets/` pipeline (and public-name dedup) as `?url`.
fn css_assets_to_emits(assets: Vec<crate::css::CssAsset>) -> Vec<AssetEmit> {
    assets
        .into_iter()
        .map(|asset| AssetEmit {
            source: asset.source,
            public_name: asset.public_name,
            tailwind_source: None,
            image_variants: None,
        })
        .collect()
}

/// The resolver specifier for one CSS `@import` edge: the target as written for
/// a plain import, or the target with the media query folded into a `?media=`
/// loader query for a media-qualified one.
fn css_import_specifier(import: &crate::css::CssImport) -> String {
    match &import.media {
        None => import.specifier.clone(),
        Some(media) => format!("{}?media={media}", import.specifier),
    }
}

/// The dependency demand for a stylesheet-to-stylesheet `@import` edge. The
/// target exports nothing; `all` keeps the edge (and the target's CSS) alive.
fn css_import_demand(specifier: String) -> DependencyDemand {
    DependencyDemand {
        specifier,
        all: true,
        names: Vec::new(),
        dynamic: false,
        optional: false,
        // A CSS `@import` is not a `require(...)`; it resolves under the same
        // conditions an ESM import does (which is also what PostCSS and Vite do).
        require_syntax: false,
        import_syntax: true,
        // A stylesheet `@import` is evaluated in place, like a static import.
        eager: true,
    }
}

/// Whether `name` can be a JavaScript named export (`export const <name> = …`):
/// a valid identifier that is not a reserved word.
fn is_valid_js_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_' || first == '$') {
        return false;
    }
    if !chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$') {
        return false;
    }
    !matches!(
        name,
        "break"
            | "case"
            | "catch"
            | "class"
            | "const"
            | "continue"
            | "debugger"
            | "default"
            | "delete"
            | "do"
            | "else"
            | "enum"
            | "export"
            | "extends"
            | "false"
            | "finally"
            | "for"
            | "function"
            | "if"
            | "import"
            | "in"
            | "instanceof"
            | "new"
            | "null"
            | "return"
            | "super"
            | "switch"
            | "this"
            | "throw"
            | "true"
            | "try"
            | "typeof"
            | "var"
            | "void"
            | "while"
            | "with"
            | "yield"
            | "let"
            | "static"
            | "await"
            | "implements"
            | "interface"
            | "package"
            | "private"
            | "protected"
            | "public"
            | "arguments"
            | "eval"
    )
}

/// A CSS Module (`*.module.css`) imported from JavaScript: the stylesheet is
/// scoped (deterministic `_<local>_<hash>` names) and joins the emitted CSS
/// concatenation, and the module's JavaScript is a synthesized mapping —
/// matching Vite's default CSS Modules behavior: the DEFAULT export is the
/// `original local -> scoped name(s)` object, plus a named export for every
/// mapping key that is a valid JS identifier. Cross-file `composes` become real
/// import edges resolved from the other module's mapping AT RUNTIME (with an
/// explicit throw if the composed name is not exported — never a silent
/// `undefined`), so editing the composed file invalidates through the ordinary
/// module graph without re-deriving the composer.
fn load_css_module(
    path: &Path,
    target: Target,
    postcss: Option<&crate::postcss::Postcss>,
) -> Result<SpecialModule, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    load_css_module_from_text(path, &text, Vec::new(), target, postcss)
}

/// The body of [`load_css_module`], parameterized over the stylesheet text so
/// a compiled-from-Sass module reuses the identical scoping/mapping pipeline.
/// `extra_source_files` are additional physical files the text was derived
/// from (Sass partials); they join `css_source_files` for invalidation.
fn load_css_module_from_text(
    path: &Path,
    text: &str,
    extra_source_files: Vec<PathBuf>,
    target: Target,
    postcss: Option<&crate::postcss::Postcss>,
) -> Result<SpecialModule, String> {
    let prefixed = maybe_postcss(text, path, postcss)?;
    let processed = crate::css::process_css_module(path, &prefixed)?;
    let mut js = String::new();
    for (index, specifier) in processed.compose_imports.iter().enumerate() {
        js.push_str(&format!(
            "import __composed_{index} from {};\n",
            quote(specifier)
        ));
    }
    let has_foreign = processed.mapping.iter().any(|(_, segments)| {
        segments
            .iter()
            .any(|segment| matches!(segment, crate::css::MappingSegment::Foreign { .. }))
    });
    if has_foreign {
        js.push_str(concat!(
            "const __compose = (mapping, name, from) => {\n",
            "  const value = mapping[name];\n",
            "  if (value === undefined) {\n",
            "    throw new Error(\"composes target \\\"\" + name + \"\\\" is not exported by \" + from);\n",
            "  }\n",
            "  return value;\n",
            "};\n",
        ));
    }
    js.push_str("const __styles = {\n");
    for (name, segments) in &processed.mapping {
        let mut parts: Vec<String> = Vec::new();
        let mut literal_run: Option<String> = None;
        for segment in segments {
            match segment {
                crate::css::MappingSegment::Literal(literal) => match &mut literal_run {
                    Some(run) => {
                        run.push(' ');
                        run.push_str(literal);
                    }
                    None => literal_run = Some(literal.clone()),
                },
                crate::css::MappingSegment::Foreign { import, name } => {
                    if let Some(run) = literal_run.take() {
                        parts.push(quote(&run));
                    }
                    parts.push(format!(
                        "__compose(__composed_{import}, {}, {})",
                        quote(name),
                        quote(&processed.compose_imports[*import])
                    ));
                }
            }
        }
        if let Some(run) = literal_run.take() {
            parts.push(quote(&run));
        }
        js.push_str(&format!(
            "  {}: {},\n",
            quote(name),
            parts.join(" + \" \" + ")
        ));
    }
    js.push_str("};\nexport default __styles;\n");
    for (name, _) in &processed.mapping {
        if is_valid_js_identifier(name) {
            js.push_str(&format!("export const {name} = __styles[{}];\n", quote(name)));
        }
    }
    let transformed = transform_module(Path::new("diffpack-css-module.js"), &js, target);
    // The synthesized mapping module's own imports (the cross-file composes)
    // plus the stylesheet's `@import` edges all become real graph dependencies,
    // resolved relative to the CSS file.
    let mut dependency_specifiers = transformed.dependencies;
    let mut dependency_demands = transformed.dependency_demands;
    for import in &processed.imports {
        let specifier = css_import_specifier(import);
        if !dependency_specifiers.contains(&specifier) {
            dependency_specifiers.push(specifier.clone());
            dependency_demands.push(css_import_demand(specifier));
        }
    }
    // The module's identity folds in everything it emits: the scoped CSS, the
    // mapping JavaScript, and the hoisted external imports.
    let mut identity = processed.css.clone();
    identity.push('\0');
    identity.push_str(&transformed.code);
    for external in &processed.external_imports {
        identity.push('\0');
        identity.push_str(external);
    }
    let mut css_source_files = processed.inlined_files;
    css_source_files.extend(extra_source_files);
    push_postcss_config(&mut css_source_files, postcss);
    Ok(SpecialModule {
        hash: content_hash(identity.as_bytes()),
        code: transformed.code,
        flat_module: transformed.flat_module,
        assets: css_assets_to_emits(processed.assets),
        css: Some(processed.css),
        css_source_files,
        css_external_imports: processed.external_imports,
        dependency_specifiers,
        dependency_demands,
    })
}

/// Runs the app's PostCSS over `text` when a config was discovered, otherwise
/// returns the text unchanged. The shared choke point every CSS-producing loader
/// funnels through (plain CSS, Sass, Less, Stylus), so a project's PostCSS runs
/// on every stylesheet exactly once per content, before the native pipeline
/// extracts `@import`s, rebases `url(...)`s, and scopes CSS Modules.
fn maybe_postcss<'a>(
    text: &'a str,
    path: &Path,
    postcss: Option<&crate::postcss::Postcss>,
) -> Result<std::borrow::Cow<'a, str>, String> {
    match postcss {
        Some(postcss) => Ok(std::borrow::Cow::Owned(postcss.process(text, path)?)),
        None => Ok(std::borrow::Cow::Borrowed(text)),
    }
}

/// Records the PostCSS config file as a build input for the module, so editing
/// the config re-derives the stylesheet (dev-server invalidation).
fn push_postcss_config(files: &mut Vec<PathBuf>, postcss: Option<&crate::postcss::Postcss>) {
    if let Some(postcss) = postcss {
        let config = postcss.config_file().to_path_buf();
        if !files.contains(&config) {
            files.push(config);
        }
    }
}

/// A `?tsr-split=<target>` virtual module: the route property extracted from the
/// original route file, re-exported under its canonical name. Loaded lazily via
/// the reference file's `import()`, so it lands in its own chunk.
fn synthesize_tsr_split(
    resource: &ResourceId,
    target: Target,
) -> Result<SpecialModule, String> {
    // The query is `tsr-split=<property>`; it selects which property was split
    // out (only `component` is implemented natively today).
    let split_property = resource
        .query
        .as_deref()
        .and_then(|query| query.split_once('='))
        .map(|(_, value)| value)
        .unwrap_or("");
    let path = Path::new(&resource.path);
    let source = fs::read_to_string(path)
        .map_err(|error| format!("cannot read route file {}: {error}", path.display()))?;
    let module_source = crate::route_split::build_split_module(path, &source, split_property)?;
    let transformed = transform_module(path, &module_source, target);
    Ok(SpecialModule {
        hash: content_hash(transformed.code.as_bytes()),
        code: transformed.code,
        flat_module: transformed.flat_module,
        assets: Vec::new(),
        css: None,
        css_source_files: Vec::new(),
        css_external_imports: Vec::new(),
        // The split module imports React and the route's own module-level deps;
        // carry them so the load paths resolve them into real graph edges.
        dependency_specifiers: transformed.dependencies,
        dependency_demands: transformed.dependency_demands,
    })
}

/// A build-generated virtual module: the given source, run through the real
/// transformer so it yields flat-linker code and export metadata like any
/// hand-written module. Used for the natively generated `tanstack-start-manifest:v`.
fn synthesize_virtual_module(source: &str) -> Result<SpecialModule, String> {
    let transformed = transform_module(Path::new("diffpack-virtual-module.js"), source, Target::Server);
    Ok(SpecialModule {
        hash: content_hash(transformed.code.as_bytes()),
        code: transformed.code,
        flat_module: transformed.flat_module,
        assets: Vec::new(),
        css: None,
        css_source_files: Vec::new(),
        css_external_imports: Vec::new(),
        // A virtual module may itself import real modules — the native server-fn
        // resolver dynamically `import()`s each server-fn module by absolute path.
        // Those specifiers MUST become graph edges (like a `?tsr-split` module's),
        // or their lowered `__dynamic(require, …)` calls have no runtime map entry
        // and fall through to a raw Node import of the untransformed source. The
        // start-manifest virtual module imports nothing, so this is empty for it.
        dependency_specifiers: transformed.dependencies,
        dependency_demands: transformed.dependency_demands,
    })
}


/// Vite's SVG inlining: UTF-8 SVGs become a compact percent-encoded
/// `data:image/svg+xml,...` (double quotes swapped for single, whitespace
/// collapsed, only the URL-hostile characters escaped, lowercase hex) — the
/// same transform Vite applies, so the emitted attribute matches its output
/// byte-for-byte. An SVG with mixed nested quotes (where the swap would change
/// meaning) or non-UTF-8 bytes falls back to base64, as Vite does.
fn svg_data_url(path: &Path, bytes: &[u8]) -> Option<String> {
    if path.extension().and_then(|value| value.to_str()) != Some("svg") {
        return None;
    }
    let text = std::str::from_utf8(bytes).ok()?;
    if text.contains('\'') && text.contains('"') {
        return None;
    }
    let collapsed = text.replace('"', "'");
    let collapsed = collapsed.split_whitespace().collect::<Vec<_>>().join(" ");
    // Whitespace between tags is structure, not content — Vite drops it.
    let collapsed = collapsed.replace("> <", "><");
    let mut out = String::with_capacity(collapsed.len() + 32);
    out.push_str("data:image/svg+xml,");
    for c in collapsed.chars() {
        match c {
            '%' => out.push_str("%25"),
            '#' => out.push_str("%23"),
            '<' => out.push_str("%3c"),
            '>' => out.push_str("%3e"),
            ' ' => out.push_str("%20"),
            '{' => out.push_str("%7b"),
            '}' => out.push_str("%7d"),
            '|' => out.push_str("%7c"),
            '^' => out.push_str("%5e"),
            '`' => out.push_str("%60"),
            '"' => out.push_str("%22"),
            '[' => out.push_str("%5b"),
            ']' => out.push_str("%5d"),
            '\\' => out.push_str("%5c"),
            '?' => out.push_str("%3f"),
            other => out.push(other),
        }
    }
    Some(out)
}

/// The MIME type an inlined asset's `data:` URI declares, by extension.
fn asset_mime_type(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
        .as_deref()
    {
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("avif") => "image/avif",
        Some("bmp") => "image/bmp",
        Some("ico") => "image/x-icon",
        Some("ttf") => "font/ttf",
        Some("otf") => "font/otf",
        Some("woff") => "font/woff",
        Some("woff2") => "font/woff2",
        Some("wasm") => "application/wasm",
        _ => "application/octet-stream",
    }
}

/// Standard base64 (RFC 4648, with padding). Hand-rolled: a data-URI encoder
/// is 20 lines, not a dependency.
fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[(triple >> 18) as usize & 63] as char);
        out.push(TABLE[(triple >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { TABLE[(triple >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { TABLE[triple as usize & 63] as char } else { '=' });
    }
    out
}

/// A content-hashed asset module: copies the file into `assets/` and exports its
/// public URL as the default export. Used for both `?url` and default asset
/// imports (images, fonts, SVG, ...).
///
/// Under [`ImageImportShape::NextObject`] a decodable PNG/JPEG default import
/// materializes as Next's static-image object (`{ src, width, height,
/// blurDataURL, variants }`) with build-emitted responsive variants, instead of a
/// bare URL string. Every other shape/format keeps the bare-URL behavior
/// byte-for-byte, so Vite/TanStack/generic builds are unaffected.
fn synthesize_asset_url(
    source_path: PathBuf,
    base: &str,
    inline_limit: usize,
    image_shape: ImageImportShape,
) -> Result<SpecialModule, String> {
    let bytes = fs::read(&source_path)
        .map_err(|error| format!("cannot read asset {}: {error}", source_path.display()))?;
    // A Tailwind v4 CSS entry imported for its URL must be compiled natively at
    // emit time, not copied verbatim: a raw copy leaves `@import 'tailwindcss'`
    // in the served file, which the browser fetches and 404s on. Capture the
    // source here (the class candidates it compiles against are only known once
    // the reachable graph is built) and mark it for the emit step.
    // Its own `@import`s are spliced in here for the same reason the global
    // stylesheet loader splices them: an imported file's Tailwind directives
    // configure this compile.
    let mut tailwind_imported_assets = Vec::new();
    let mut tailwind_inlined_files = Vec::new();
    let tailwind_source = if is_css_path(&source_path) {
        let text = String::from_utf8_lossy(&bytes);
        if crate::tailwind::needs_native_tailwind_compile(&text) {
            let entry = crate::css::inline_tailwind_entry(&source_path, &text)?;
            tailwind_imported_assets = entry.assets;
            tailwind_inlined_files = entry.inlined_files;
            Some(entry.css)
        } else {
            None
        }
    } else {
        None
    };
    // The bytes a Tailwind entry SERVES are the compiled stylesheet, not these
    // source bytes, so the app's resolved theme (which varies with the installed
    // `tailwindcss`) must be in the content hash: otherwise upgrading Tailwind
    // changes the stylesheet's content while leaving its immutable-cached URL
    // identical. The scanned class candidates are the compile's other input and
    // are still NOT hashed — they are unknown until the graph is built, so a
    // pure class-set change reuses the URL (pre-existing, tracked separately).
    let public_name = if tailwind_source.is_some() {
        let mut hashed = bytes.clone();
        if let Some(theme) = app_tailwind_theme(&source_path) {
            hashed.extend_from_slice(theme.as_bytes());
        }
        asset_public_name(&source_path, content_hash(&hashed))
    } else {
        asset_public_name(&source_path, content_hash(&bytes))
    };
    // Next static-image import: a decodable PNG/JPEG becomes the object shape with
    // responsive variants + an auto blurDataURL. Runs BEFORE the inline-limit
    // branch so a small raster is never inlined away (the object shape needs a
    // real emitted file + variants). Undecodable rasters (or non-png/jpeg formats
    // the `image` crate here can't decode) fall through to the bare-URL path.
    if let ImageImportShape::NextObject { responsive_variants } = image_shape
        && let Some(module) = synthesize_next_image_object(
            &source_path,
            &bytes,
            &public_name,
            base,
            responsive_variants,
        )?
    {
        return Ok(module);
    }
    // A plain ES module exporting the asset URL, run through the real transformer
    // so it yields flat-linker code and export metadata like any hand-written
    // module.
    // Vite's `assetsInlineLimit`: a small asset becomes a `data:` URI instead
    // of an emitted file — one request fewer, byte-parity with Vite's output
    // model. A Tailwind entry is exempt: its bytes here are the RAW source,
    // and the emit step replaces them with the compiled stylesheet.
    if inline_limit > 0 && bytes.len() <= inline_limit && tailwind_source.is_none() {
        let data_uri = svg_data_url(&source_path, &bytes).unwrap_or_else(|| {
            format!(
                "data:{};base64,{}",
                asset_mime_type(&source_path),
                base64_encode(&bytes)
            )
        });
        let synthetic = format!("export default {};\n", quote(&data_uri));
        let transformed =
            transform_module(Path::new("diffpack-url-asset.js"), &synthetic, Target::Server);
        return Ok(SpecialModule {
            hash: content_hash(transformed.code.as_bytes()),
            code: transformed.code,
            flat_module: transformed.flat_module,
            assets: Vec::new(),
            css: None,
            css_source_files: Vec::new(),
            css_external_imports: Vec::new(),
            dependency_specifiers: Vec::new(),
            dependency_demands: Vec::new(),
        });
    }
    let synthetic = format!("export default {};\n", quote(&format!("{base}assets/{public_name}")));
    let transformed = transform_module(Path::new("diffpack-url-asset.js"), &synthetic, Target::Server);
    let mut assets = vec![AssetEmit {
        source: source_path,
        public_name,
        tailwind_source,
        image_variants: None,
    }];
    assets.extend(css_assets_to_emits(tailwind_imported_assets));
    Ok(SpecialModule {
        hash: content_hash(transformed.code.as_bytes()),
        code: transformed.code,
        flat_module: transformed.flat_module,
        assets,
        css: None,
        css_source_files: tailwind_inlined_files,
        css_external_imports: Vec::new(),
        dependency_specifiers: Vec::new(),
        dependency_demands: Vec::new(),
    })
}

/// The public filename for one responsive variant of a content-hashed image
/// asset: `<stem>-<width>.<ext>` derived from the original's `public_name`
/// (e.g. `shot-1a2b3c4d.png` + 640 -> `shot-1a2b3c4d-640.png`). Deterministic so
/// the object's `variants` map (written here) and the emitted files (written in
/// [`ModuleGraph::emit_assets`]) agree without shared state.
fn asset_variant_public_name(public_name: &str, width: u32) -> String {
    match public_name.rsplit_once('.') {
        Some((stem, ext)) => format!("{stem}-{width}.{ext}"),
        None => format!("{public_name}-{width}"),
    }
}

/// Build Next's static-image object module for a default raster import, or `None`
/// if the bytes are not a PNG/JPEG this build can decode (caller then falls back
/// to the bare-URL string). Decodes the source once for intrinsic dimensions, a
/// tiny blurDataURL, and the responsive-variant plan; the variants themselves are
/// emitted at build time by [`ModuleGraph::emit_assets`] (NO image server).
fn synthesize_next_image_object(
    source_path: &Path,
    bytes: &[u8],
    public_name: &str,
    base: &str,
    responsive_variants: bool,
) -> Result<Option<SpecialModule>, String> {
    let ext = source_path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase());
    let is_png_or_jpeg = matches!(ext.as_deref(), Some("png" | "jpg" | "jpeg"));
    if !is_png_or_jpeg {
        return Ok(None);
    }
    // Decode from the bytes already in hand (no second filesystem read).
    let Ok(decoded) = image::load_from_memory(bytes) else {
        // An undecodable/corrupt raster: fall back to the bare-URL string rather
        // than throwing — the shim then renders it unoptimized, honest passthrough.
        return Ok(None);
    };
    let width = decoded.width();
    let height = decoded.height();
    if width == 0 || height == 0 {
        return Ok(None);
    }
    let out_ext = if ext.as_deref() == Some("jpg") { "jpeg" } else { ext.as_deref().unwrap_or("png") };
    let blur = generate_blur_data_url(&decoded, out_ext)?;
    let src_url = format!("{base}assets/{public_name}");
    // With optimization off the object OMITS `variants` entirely (not an empty map —
    // `{}` is truthy, and the shim's "no variants = render raw" test reads it as a
    // value). Next's own static import behaves the same way under `images.unoptimized`:
    // the `<img>` gets the full-resolution `src` and no `srcset`.
    let widths = responsive_variants.then(|| crate::next_adapter::variant_widths(width));
    let variants_field = match &widths {
        Some(widths) => {
            let variants_js = widths
                .iter()
                .map(|&w| {
                    let url = format!("{base}assets/{}", asset_variant_public_name(public_name, w));
                    format!("{}: {}", quote(&w.to_string()), quote(&url))
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!(", variants: {{ {variants_js} }}")
        }
        None => String::new(),
    };
    let synthetic = format!(
        "export default {{ src: {}, width: {width}, height: {height}, blurDataURL: {}{variants_field} }};\n",
        quote(&src_url),
        quote(&blur),
    );
    let transformed = transform_module(Path::new("diffpack-image-import.js"), &synthetic, Target::Server);
    Ok(Some(SpecialModule {
        hash: content_hash(transformed.code.as_bytes()),
        code: transformed.code,
        flat_module: transformed.flat_module,
        assets: vec![AssetEmit {
            source: source_path.to_path_buf(),
            public_name: public_name.to_string(),
            tailwind_source: None,
            image_variants: widths,
        }],
        css: None,
        css_source_files: Vec::new(),
        css_external_imports: Vec::new(),
        dependency_specifiers: Vec::new(),
        dependency_demands: Vec::new(),
    }))
}

/// Encode a tiny (~8px-wide) downscale of `img` as a base64 `data:` URI — the
/// `blurDataURL` for `placeholder="blur"`. PNG sources keep PNG (transparency);
/// JPEG sources keep JPEG. Generated natively via the already-vendored `image`
/// crate (no sharp/squoosh dependency, no image server).
pub(crate) fn generate_blur_data_url(img: &image::DynamicImage, ext: &str) -> Result<String, String> {
    const BLUR_WIDTH: u32 = 8;
    let (w, h) = (img.width().max(1), img.height().max(1));
    let target_h = ((h as u64 * BLUR_WIDTH as u64) / w as u64).max(1) as u32;
    let small = img.resize_exact(BLUR_WIDTH, target_h, image::imageops::FilterType::Triangle);
    let (format, mime) = if ext == "jpeg" || ext == "jpg" {
        (image::ImageFormat::Jpeg, "image/jpeg")
    } else {
        (image::ImageFormat::Png, "image/png")
    };
    let mut buffer = std::io::Cursor::new(Vec::new());
    small
        .write_to(&mut buffer, format)
        .map_err(|error| format!("cannot encode blur placeholder: {error}"))?;
    Ok(format!("data:{mime};base64,{}", base64_encode(&buffer.into_inner())))
}

/// A `?raw` module: the file's contents inlined as the default string export.
fn synthesize_raw(source_path: &Path) -> Result<SpecialModule, String> {
    let text = fs::read_to_string(source_path)
        .map_err(|error| format!("cannot read {}: {error}", source_path.display()))?;
    let synthetic = format!("export default {};\n", quote(&text));
    let transformed = transform_module(Path::new("diffpack-raw-asset.js"), &synthetic, Target::Server);
    Ok(SpecialModule {
        hash: content_hash(transformed.code.as_bytes()),
        code: transformed.code,
        flat_module: transformed.flat_module,
        assets: Vec::new(),
        css: None,
        css_source_files: Vec::new(),
        css_external_imports: Vec::new(),
        dependency_specifiers: Vec::new(),
        dependency_demands: Vec::new(),
    })
}

/// A global stylesheet import: an empty JavaScript module (the import has no
/// bindings) whose text is extracted into the output stylesheet. Its top-level
/// `@import`s become real graph dependency edges (the imported file's CSS is
/// emitted before the importer's, deduped once per graph) and its relative
/// `url(...)`s are rewritten to content-hashed public asset URLs.
fn load_stylesheet(
    path: &Path,
    postcss: Option<&crate::postcss::Postcss>,
) -> Result<SpecialModule, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    // A Tailwind v4 entry's `@import 'tailwindcss'` is a COMPILER invocation,
    // not a stylesheet import; resolving it through the module resolver would
    // inline the wrong thing. Carry the RAW text as the module's stylesheet —
    // `emit_css` detects it and compiles through the native Tailwind engine at
    // emit time (exactly like the `?url` asset path), so class candidates are
    // re-scanned on every emit and the compile stays off the per-edit
    // transform hot path.
    if crate::tailwind::needs_native_tailwind_compile(&text) {
        // The entry's OTHER `@import`s are spliced in first: an imported file's
        // `@theme`/`@plugin`/`@source`/`@utility` configures the same compile,
        // so the whole graph must be one text before the compiler sees it.
        let entry = crate::css::inline_tailwind_entry(path, &text)?;
        return Ok(SpecialModule {
            hash: content_hash(entry.css.as_bytes()),
            code: String::new(),
            flat_module: None,
            assets: css_assets_to_emits(entry.assets),
            css: Some(entry.css),
            css_source_files: entry.inlined_files,
            css_external_imports: entry.external_imports,
            dependency_specifiers: Vec::new(),
            dependency_demands: Vec::new(),
        });
    }
    // (Legacy Tailwind v3 `@tailwind base/components/utilities` entries are captured by
    // the `needs_native_tailwind_compile` gate above and compiled natively through the
    // v4 pipeline — the directives expand to the same base/components/utilities layers.)
    load_stylesheet_from_text(path, &text, Vec::new(), postcss)
}

/// The body of [`load_stylesheet`], parameterized over the stylesheet text so
/// a compiled-from-Sass global stylesheet reuses the identical import/url
/// pipeline. `extra_source_files` are additional physical files the text was
/// derived from (Sass partials); they join `css_source_files` for
/// invalidation.
fn load_stylesheet_from_text(
    path: &Path,
    text: &str,
    extra_source_files: Vec<PathBuf>,
    postcss: Option<&crate::postcss::Postcss>,
) -> Result<SpecialModule, String> {
    let prefixed = maybe_postcss(text, path, postcss)?;
    let processed = crate::css::process_global_css(path, &prefixed)?;
    let mut identity = processed.css.clone();
    for external in &processed.external_imports {
        identity.push('\0');
        identity.push_str(external);
    }
    let dependency_specifiers = processed
        .imports
        .iter()
        .map(css_import_specifier)
        .collect::<Vec<_>>();
    let dependency_demands = dependency_specifiers
        .iter()
        .cloned()
        .map(css_import_demand)
        .collect();
    let mut css_source_files = processed.inlined_files;
    css_source_files.extend(extra_source_files);
    push_postcss_config(&mut css_source_files, postcss);
    Ok(SpecialModule {
        hash: content_hash(identity.as_bytes()),
        code: String::new(),
        flat_module: None,
        assets: css_assets_to_emits(processed.assets),
        css: Some(processed.css),
        css_source_files,
        css_external_imports: processed.external_imports,
        dependency_specifiers,
        dependency_demands,
    })
}

/// Whether a resolved path is a plain global stylesheet (`import "./app.css"`).
fn is_css_path(path: &Path) -> bool {
    matches!(path.extension().and_then(|value| value.to_str()), Some("css"))
}

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

fn tailwind_scan_root(css_path: &Path, source_css: &str) -> PathBuf {
    let css_dir = css_path.parent().unwrap_or_else(|| Path::new("."));
    tailwind_source_root(source_css)
        .map(|rel| css_dir.join(rel))
        .unwrap_or_else(|| {
            let mut root = css_dir;
            for ancestor in css_dir.ancestors() {
                if ancestor.join("package.json").is_file() {
                    root = ancestor;
                    break;
                }
            }
            root.to_path_buf()
        })
}

/// The installed `tailwindcss` package directory Node resolution reaches from a
/// stylesheet: the nearest ancestor of the STYLESHEET holding a
/// `node_modules/tailwindcss/theme.css`.
///
/// Anchored on the stylesheet, not on the candidate scan root. Module resolution
/// is defined against the importing file; a `source(...)` scan root is a
/// source-tree concept with no relation to it, and joining `node_modules` onto
/// it only found the install when the two happened to coincide — TanStack
/// Start's `src/styles/app.css` with `source('../')` scans `src/`, which holds
/// no `node_modules`, so every such app silently compiled against the vendored
/// theme and shipped a stale `--font-sans`. Walking up from the file is also
/// what makes pnpm's nested layout and a monorepo root install resolve.
fn installed_tailwind_dir(css_path: &Path) -> Option<PathBuf> {
    css_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .ancestors()
        .map(|dir| dir.join("node_modules/tailwindcss"))
        .find(|package| package.join("theme.css").is_file())
}

/// The app's own installed Tailwind default theme, when present. Compiling
/// against it matches the exact Tailwind version the reference build used
/// (default tokens like `--font-sans` changed between v4 releases); without
/// it the vendored copy in `src/tailwind_theme.css` applies.
fn app_tailwind_theme(css_path: &Path) -> Option<String> {
    let package = installed_tailwind_dir(css_path)?;
    fs::read_to_string(package.join("theme.css")).ok()
}

/// States which engine produced one Tailwind entry's stylesheet, once per
/// (entry, message). A build that compiles the same entry for several passes
/// (client + react-server) says it once.
///
/// Silence here is not an option: whether a sheet came from diffpack's native
/// engine or from the app's own `tailwindcss` decides what is in it, and a reader
/// who has to infer that from a pixel diff has been failed.
fn report_tailwind_engine(css_path: &Path, message: &str) {
    static REPORTED: Mutex<Option<BTreeSet<String>>> = Mutex::new(None);
    let line = format!("[tailwind] {}: {message}", css_path.display());
    let mut reported = REPORTED.lock().unwrap();
    if reported.get_or_insert_with(BTreeSet::new).insert(line.clone()) {
        eprintln!("{line}");
    }
}

/// The `version` field of an installed package's `package.json`.
fn installed_package_version(package: &Path) -> Option<String> {
    let manifest = fs::read_to_string(package.join("package.json")).ok()?;
    let value: serde_json::Value = serde_json::from_str(&manifest).ok()?;
    value.get("version")?.as_str().map(str::to_string)
}

/// Warns, once per differing version, when the app's installed `tailwindcss` is
/// not the release the vendored data came from. The installed `theme.css` is
/// still used (its tokens are what the app's own build would emit), but the
/// preflight and the version banner remain the vendored ones — a mixture that
/// exists in no released Tailwind, so it is stated rather than left to be
/// discovered as a pixel diff.
fn warn_on_tailwind_version_drift(package: &Path) {
    static WARNED: Mutex<Option<BTreeSet<String>>> = Mutex::new(None);
    let Some(installed) = installed_package_version(package) else {
        return;
    };
    if installed == crate::tailwind::VERSION {
        return;
    }
    let mut warned = WARNED.lock().unwrap();
    if !warned.get_or_insert_with(BTreeSet::new).insert(installed.clone()) {
        return;
    }
    eprintln!(
        "warning: {} is tailwindcss v{installed}, but diffpack's vendored Tailwind data is \
         v{}. Its theme tokens are used as installed; the preflight and version banner \
         remain v{}. Re-vendor src/tailwind_theme.css / src/tailwind_preflight*.css if the \
         output diverges.",
        package.display(),
        crate::tailwind::VERSION,
        crate::tailwind::VERSION,
    );
}

/// The full app theme fed to the Tailwind compiler: the installed `tailwindcss`
/// default `theme.css`, EXTENDED with the `@theme`/`@keyframes` tokens derived from a
/// legacy JS config referenced by a `@config '<path>'` directive in `css` (if any).
/// A `@config`-defined token overrides the default (it is appended after it).
fn app_tailwind_theme_full(scan_root: &Path, css: &str, css_path: &Path) -> Option<String> {
    let base = app_tailwind_theme(css_path);
    let config = at_config_theme(scan_root, css, css_path);
    match (base, config) {
        (Some(base), Some(cfg)) => Some(format!("{base}\n{cfg}")),
        (base, None) => base,
        // A v3 config with no installed `tailwindcss/theme.css`: merge the config tokens
        // ON TOP of the vendored default theme so the config EXTENDS the default scale
        // rather than replacing it (a bare `--color-brand` must not drop `p-4`/`flex`).
        (None, Some(cfg)) => Some(format!("{}\n{}", crate::tailwind::vendored_theme_css(), cfg)),
    }
}

/// The path string in a `@config '<path>'` / `@config "<path>"` directive, if present.
fn parse_at_config(css: &str) -> Option<String> {
    let after = &css[css.find("@config")? + "@config".len()..];
    let open = after.find(['\'', '"'])?;
    let quote = after.as_bytes()[open] as char;
    let inner = &after[open + 1..];
    let close = inner.find(quote)?;
    Some(inner[..close].to_string())
}

/// Evaluate a `@config`-referenced legacy JS Tailwind config (via node + the app's
/// own jiti) into v4 `@theme`/`@keyframes` CSS. Returns `None` when there is no
/// `@config`, the config file is missing, or node is unavailable — the compile then
/// proceeds on the default theme (a `@config` on a config with only content/plugins
/// contributes no theme tokens anyway). Never silently mis-maps: the node evaluator
/// reports unmapped theme categories on stderr, surfaced here.
/// Discovers a legacy v3 `tailwind.config.{js,cjs,mjs,ts}` at the project scan root
/// (v3 apps declare the config there, with no `@config` directive in the CSS). Returns
/// the first that exists.
fn discover_v3_config(scan_root: &Path) -> Option<PathBuf> {
    ["tailwind.config.js", "tailwind.config.cjs", "tailwind.config.mjs", "tailwind.config.ts"]
        .iter()
        .map(|name| scan_root.join(name))
        .find(|p| p.exists())
}

fn at_config_theme(scan_root: &Path, css: &str, css_path: &Path) -> Option<String> {
    // A `@config '<path>'` directive names the config explicitly (v4-style). Otherwise a
    // legacy v3 entry auto-discovers `tailwind.config.*` at the scan root — but a v4
    // entry with no `@config` uses NO JS config (so a stray tailwind.config.js is not
    // picked up for it).
    let config_path = match parse_at_config(css) {
        Some(rel) => css_path.parent()?.join(rel),
        None if crate::tailwind::is_tailwind_v3_entry(css) => discover_v3_config(scan_root)?,
        None => return None,
    };
    if !config_path.exists() {
        eprintln!(
            "[tailwind @config] config file not found: {} (theme tokens from it will be missing)",
            config_path.display()
        );
        return None;
    }
    // The evaluator resolves jiti + the config's imports from the CONFIG's
    // node_modules, so it can live in a temp file; run it from the config's dir.
    let loader = std::env::temp_dir().join("diffpack-tailwind-config-eval.mjs");
    if fs::write(&loader, include_str!("../scripts/tailwind-config-eval.mjs")).is_err() {
        return None;
    }
    let output = std::process::Command::new("node")
        .arg(&loader)
        .arg(&config_path)
        .current_dir(config_path.parent().unwrap_or_else(|| Path::new(".")))
        .output()
        .ok()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.trim().is_empty() {
        eprintln!("[tailwind @config] {}", stderr.trim());
    }
    if !output.status.success() {
        return None;
    }
    let theme = String::from_utf8_lossy(&output.stdout).to_string();
    (!theme.trim().is_empty()).then_some(theme)
}

/// Parses the `source('...')` argument of a Tailwind v4 `@import 'tailwindcss'`
/// entry: the (entry-relative) directory the compiler scans for classes.
fn tailwind_source_root(source_css: &str) -> Option<String> {
    let start = source_css.find("source(")? + "source(".len();
    let rest = &source_css[start..];
    let end = rest.find(')')?;
    Some(rest[..end].trim().trim_matches(['\'', '"']).to_string())
}

/// What the candidate scan must not descend into: the scan root's `.gitignore`
/// entries (Tailwind's own scanner respects `.gitignore`, which is how a
/// checked-in reference build never picks candidates out of `dist/`) plus this
/// build's own output directory (never scan what we emitted).
struct ScanSkip {
    /// Simple ignored names (`dist`, `logs`): skipped wherever they appear.
    names: Vec<String>,
    /// Ignored filename suffixes from `*.<ext>`-style patterns (`.log`).
    suffixes: Vec<String>,
    /// The build's canonical output root.
    out_root: Option<PathBuf>,
    /// `@source not "<glob>"` exclusions, brace-expanded and split into
    /// segments. A path matching any of them is skipped wherever the scan
    /// reaches it.
    excluded: Vec<Vec<String>>,
}

impl ScanSkip {
    fn for_root(scan_root: &Path, out_root: &Path) -> ScanSkip {
        let mut names = Vec::new();
        let mut suffixes = Vec::new();
        if let Ok(gitignore) = fs::read_to_string(scan_root.join(".gitignore")) {
            for line in gitignore.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') || line.starts_with('!') {
                    continue;
                }
                let entry = line.trim_matches('/');
                if let Some(suffix) = entry.strip_prefix("*.") {
                    if !suffix.contains(['*', '/', '?', '[']) {
                        suffixes.push(format!(".{suffix}"));
                    }
                } else if !entry.contains(['*', '/', '?', '[']) {
                    names.push(entry.to_string());
                }
            }
        }
        ScanSkip {
            names,
            suffixes,
            out_root: fs::canonicalize(out_root).ok(),
            excluded: Vec::new(),
        }
    }

    fn skips(&self, path: &Path, name: &str) -> bool {
        if name.starts_with('.') || name == "node_modules" {
            return true;
        }
        if self.names.iter().any(|n| n == name)
            || self.suffixes.iter().any(|s| name.ends_with(s.as_str()))
        {
            return true;
        }
        if !self.excluded.is_empty() {
            let segments = path_segments(path);
            if self
                .excluded
                .iter()
                .any(|pattern| glob_matches(pattern, &segments))
            {
                return true;
            }
        }
        if let Some(out_root) = &self.out_root
            && let Ok(canonical) = fs::canonicalize(path)
            && canonical == *out_root
        {
            return true;
        }
        false
    }
}

/// The `@source` directives of a compiled Tailwind entry, split into the extra
/// paths to scan and the `not`-negated paths to exclude. Every path is absolute
/// (`css::absolutize_source_directives` anchors each one to the file that wrote
/// it before the entry's imports are spliced together).
fn tailwind_source_globs(css: &str) -> Result<(Vec<String>, Vec<String>), String> {
    let mut included = Vec::new();
    let mut excluded = Vec::new();
    let mut rest = css;
    while let Some(at) = rest.find("@source") {
        let body = &rest[at + "@source".len()..];
        let Some(end) = body.find(';') else {
            break;
        };
        let statement = body[..end].trim();
        rest = &body[end + 1..];
        let (negated, target) = match statement.strip_prefix("not") {
            Some(tail) if tail.starts_with(char::is_whitespace) => (true, tail.trim()),
            _ => (false, statement),
        };
        let Some(path) = target
            .strip_prefix(['"', '\''])
            .and_then(|value| value.get(..value.len().saturating_sub(1)))
        else {
            return Err(format!("@source must name a quoted path (got `@source {statement}`)"));
        };
        if path.contains('[') {
            return Err(format!(
                "`@source \"{path}\"` uses a character class, which diffpack's Tailwind \
                 source matcher does not implement (it supports `**`, `*`, `?` and `{{a,b}}`)"
            ));
        }
        if negated {
            excluded.push(path.to_string());
        } else {
            included.push(path.to_string());
        }
    }
    Ok((included, excluded))
}

/// Splits a path into its components as strings, for glob matching.
fn path_segments(path: &Path) -> Vec<String> {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect()
}

/// Expands `{a,b}` alternations into the concrete patterns they stand for.
/// Nested and repeated groups expand as the product, matching shell/glob
/// semantics.
fn expand_braces(pattern: &str) -> Vec<String> {
    let Some(open) = pattern.find('{') else {
        return vec![pattern.to_string()];
    };
    let mut depth = 0usize;
    let mut close = None;
    for (offset, byte) in pattern[open..].bytes().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(open + offset);
                    break;
                }
            }
            _ => {}
        }
    }
    let Some(close) = close else {
        return vec![pattern.to_string()];
    };
    let mut alternatives = Vec::new();
    let mut depth = 0usize;
    let mut start = open + 1;
    let inner = &pattern[open + 1..close];
    for (offset, byte) in inner.bytes().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => depth -= 1,
            b',' if depth == 0 => {
                alternatives.push(&pattern[start..open + 1 + offset]);
                start = open + 1 + offset + 1;
            }
            _ => {}
        }
    }
    alternatives.push(&pattern[start..close]);
    let mut out = Vec::new();
    for alternative in alternatives {
        let expanded = format!("{}{alternative}{}", &pattern[..open], &pattern[close + 1..]);
        out.extend(expand_braces(&expanded));
    }
    out
}

/// Whether a `*`/`?` pattern segment matches one path component. `*` matches any
/// run of characters within the component (never a `/`), `?` exactly one.
fn segment_matches(pattern: &str, name: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let name: Vec<char> = name.chars().collect();
    // Classic backtracking wildcard match, iterative so a pathological pattern
    // cannot blow the stack.
    let (mut p, mut n) = (0usize, 0usize);
    let (mut star, mut backtrack) = (None, 0usize);
    while n < name.len() {
        if p < pattern.len() && (pattern[p] == '?' || pattern[p] == name[n]) {
            p += 1;
            n += 1;
        } else if p < pattern.len() && pattern[p] == '*' {
            star = Some(p);
            backtrack = n;
            p += 1;
        } else if let Some(star) = star {
            p = star + 1;
            backtrack += 1;
            n = backtrack;
        } else {
            return false;
        }
    }
    while p < pattern.len() && pattern[p] == '*' {
        p += 1;
    }
    p == pattern.len()
}

/// Whether a brace-expanded, segment-split glob matches a path's segments.
/// `**` matches any number of segments (including none).
fn glob_matches(pattern: &[String], path: &[String]) -> bool {
    if pattern.is_empty() {
        return path.is_empty();
    }
    if pattern[0] == "**" {
        // `**` consumes zero or more segments; try each split point.
        for taken in 0..=path.len() {
            if glob_matches(&pattern[1..], &path[taken..]) {
                return true;
            }
        }
        return false;
    }
    if path.is_empty() {
        return false;
    }
    segment_matches(&pattern[0], &path[0]) && glob_matches(&pattern[1..], &path[1..])
}

/// Whether a path segment carries glob metacharacters.
fn is_glob_segment(segment: &str) -> bool {
    segment.contains(['*', '?', '{'])
}

/// Reads every source file an `@source` pattern selects. A pattern with no glob
/// metacharacters names a file (read directly) or a directory (walked whole,
/// exactly as Tailwind treats a bare `@source "./dir"`).
fn collect_glob_sources(pattern: &str, out: &mut Vec<(PathBuf, String)>, skip: &ScanSkip) {
    for expanded in expand_braces(pattern) {
        let segments = path_segments(Path::new(&expanded));
        let literal = segments.iter().take_while(|s| !is_glob_segment(s)).count();
        let root: PathBuf = segments[..literal].iter().collect();
        if literal == segments.len() {
            if root.is_dir() {
                // An `@source` directory walk is not cancellable: the caller checks
                // between patterns, which is granular enough for the handful an app
                // declares.
                collect_scan_sources(&root, out, skip, &EmitCancel::never());
            } else if let Ok(source) = fs::read_to_string(&root) {
                out.push((root.clone(), source));
            }
            continue;
        }
        collect_matching_sources(&root, &segments, out, skip);
    }
}

/// Walks `directory`, reading every file whose full path matches `pattern`.
fn collect_matching_sources(
    directory: &Path,
    pattern: &[String],
    out: &mut Vec<(PathBuf, String)>,
    skip: &ScanSkip,
) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if skip.skips(&path, &name) {
            continue;
        }
        if path.is_dir() {
            collect_matching_sources(&path, pattern, out, skip);
        } else if glob_matches(pattern, &path_segments(&path))
            && let Ok(source) = fs::read_to_string(&path)
        {
            out.push((path, source));
        }
    }
}

/// Recursively gathers the sources the utility-class candidate scan reads:
/// every JS/TS/JSX/HTML file under the scan root. Skips `node_modules`,
/// dot-directories, `.gitignore`d entries (as Tailwind does), and the build's
/// own output directory, so only the app's own classes are scanned. The files
/// are scanned together (`scan_class_candidates_multi`) so identifiers resolve
/// across module boundaries.
/// Reads every scannable source under `root` into `out`. Returns false if `cancel`
/// fired part-way, in which case `out` is incomplete and must not be scanned.
fn collect_scan_sources(
    root: &Path,
    out: &mut Vec<(PathBuf, String)>,
    skip: &ScanSkip,
    cancel: &EmitCancel<'_>,
) -> bool {
    let Ok(entries) = fs::read_dir(root) else {
        return true;
    };
    for entry in entries.flatten() {
        if cancel.cancelled() {
            return false;
        }
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if skip.skips(&path, &name) {
            continue;
        }
        if path.is_dir() {
            if !collect_scan_sources(&path, out, skip, cancel) {
                return false;
            }
        } else if matches!(
            path.extension().and_then(|value| value.to_str()),
            Some("js" | "jsx" | "ts" | "tsx" | "mjs" | "cjs" | "html")
        ) && let Ok(source) = fs::read_to_string(&path)
        {
            out.push((path, source));
        }
    }
    true
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

/// A resolved file that no loader claims and that is not JavaScript either.
/// Split three ways because the honest remedy differs: diffpack either knows
/// exactly what the file is and which compiler it would need, knows it is not
/// source at all, or knows only that nothing here parses it.
enum UnhandledSource {
    /// A recognized source format whose compiler is a JS library a bundler runs
    /// as a plugin. `kind` reads as "`.astro` is {kind}, not JavaScript".
    NeedsCompiler {
        kind: &'static str,
        compiler: &'static str,
    },
    /// A prebuilt native addon (`.node`): machine code, not source.
    NativeAddon,
    /// An extension no loader claims. All diffpack knows is that it is not JS.
    NoLoader,
}

/// Source formats diffpack recognizes and deliberately cannot compile: each one
/// needs a compiler that bundlers host as a JS plugin, and diffpack hosts no JS
/// plugins (README: "Not yet: JS plugin hosting"). Naming the compiler is the
/// whole point of the table — the alternative is parsing someone else's language
/// as JavaScript and blaming the app for a syntax error it does not have.
const COMPILED_SOURCE_KINDS: &[(&str, &str, &str)] = &[
    (
        "astro",
        "an Astro component",
        "the Astro compiler (@astrojs/compiler)",
    ),
    (
        "marko",
        "a Marko template",
        "the Marko compiler (@marko/compiler)",
    ),
    (
        "riot",
        "a Riot component",
        "the Riot compiler (@riotjs/compiler)",
    ),
    ("imba", "an Imba module", "the Imba compiler"),
    (
        "civet",
        "a Civet module",
        "the Civet compiler (@danielx/civet)",
    ),
    ("coffee", "a CoffeeScript module", "the CoffeeScript compiler"),
    ("res", "a ReScript module", "the ReScript compiler"),
    ("resi", "a ReScript interface", "the ReScript compiler"),
    ("re", "a Reason module", "the Reason compiler"),
    ("rei", "a Reason interface", "the Reason compiler"),
    ("elm", "an Elm module", "the Elm compiler"),
];

/// Whether a resolved file falls outside every loader AND outside the JavaScript
/// family. [`load_special_module`] returning `None` means "read this as
/// JavaScript", so without this check "unknown extension" and "JavaScript" are
/// the same branch: oxc parses an Astro component's markup as a JS expression
/// and reports `Unexpected JSX expression`, blaming the app for diffpack's own
/// gap. The JS family is therefore an explicit allow-list, not an implicit
/// default.
fn unhandled_source(path: &Path) -> Option<UnhandledSource> {
    // No extension at all IS JavaScript: package `main` entries and `bin` scripts
    // under `node_modules` are routinely extensionless.
    let extension = path.extension().and_then(|value| value.to_str())?;
    // `.mts`/`.cts` are absent from the resolver's extension list, so they only
    // ever arrive via an explicit specifier — but they are still TypeScript.
    if matches!(
        extension,
        "js" | "jsx" | "mjs" | "cjs" | "ts" | "tsx" | "mts" | "cts" | "json" | "md" | "mdx"
    ) {
        return None;
    }
    // A `.vue`/`.svelte` single-file component IS handled: the JS load path
    // compiles it with the app's own compiler before parsing (see
    // [`precompile_component`] and [`crate::sfc`]). It must not be reported as a
    // gap here, and it must not be read as JavaScript either — the compile step
    // between the read and the parse is what makes both true.
    if crate::sfc::is_component_path(path) {
        return None;
    }
    if extension == "node" {
        return Some(UnhandledSource::NativeAddon);
    }
    match COMPILED_SOURCE_KINDS
        .iter()
        .find(|(candidate, _, _)| *candidate == extension)
    {
        Some((_, kind, compiler)) => Some(UnhandledSource::NeedsCompiler { kind, compiler }),
        None => Some(UnhandledSource::NoLoader),
    }
}

/// Renders an unhandled source: the file, what its extension actually is, what
/// compiling it would require, and the way forward. Deliberately says the file
/// was FOUND — the failure this replaces was routinely read as a missing import,
/// which sent readers hunting for a resolution problem that does not exist.
fn unhandled_source_message(path: &Path, unhandled: &UnhandledSource) -> String {
    let file = path.display();
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    let found = "the file was found on disk: this is neither a missing import nor a JavaScript \
                 syntax error";
    match unhandled {
        UnhandledSource::NeedsCompiler { kind, compiler } => format!(
            "{file}: `.{extension}` is {kind}, not JavaScript\n  \
             compiling it requires {compiler}; diffpack hosts no JS plugins and has no built-in \
             `.{extension}` compiler\n  {found}\n  \
             build this project with its own toolchain instead"
        ),
        UnhandledSource::NativeAddon => format!(
            "{file}: `.{extension}` is a prebuilt native addon, not JavaScript\n  \
             a native addon is machine code loaded by Node's `process.dlopen`, and diffpack \
             cannot put native code in a JavaScript bundle\n  {found}\n  \
             build this project with its own toolchain instead"
        ),
        UnhandledSource::NoLoader => {
            let name = path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or_default();
            format!(
                "{file}: no loader handles the `.{extension}` extension\n  \
                 diffpack loads .js/.jsx/.mjs/.cjs/.ts/.tsx/.mts/.cts, .json, .md/.mdx, \
                 .css/.scss/.sass/.less/.styl/.stylus, and static assets; nothing else is parsed \
                 as JavaScript\n  \
                 the file was found on disk: this is not a missing import\n  \
                 to import its contents or its URL, use an explicit loader query: \
                 `./{name}?raw` or `./{name}?url`"
            )
        }
    }
}

/// The content-hashed public filename for an asset, e.g. `app-1a2b3c4d5e6f7080.css`.
pub(crate) fn asset_public_name(path: &Path, hash: u64) -> String {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("asset");
    match path.extension().and_then(|value| value.to_str()) {
        Some(extension) => format!("{stem}-{hash:016x}.{extension}"),
        None => format!("{stem}-{hash:016x}"),
    }
}

/// The Node built-in module names, without the `node:` prefix. ONE list: the
/// build-time classifier ([`is_node_builtin`]) and the browser runtime's
/// `requireNative` stub (see [`browser_require_native`]) are both derived from
/// it, so the two can never disagree about what "a Node built-in" means.
pub(crate) const NODE_BUILTINS: &[&str] = &[
    "assert",
    "async_hooks",
    "buffer",
    "child_process",
    "cluster",
    "console",
    "constants",
    "crypto",
    "dgram",
    "diagnostics_channel",
    "dns",
    "domain",
    "events",
    "fs",
    "http",
    "http2",
    "https",
    "inspector",
    "module",
    "net",
    "os",
    "path",
    "perf_hooks",
    "process",
    "punycode",
    "querystring",
    "readline",
    "repl",
    "stream",
    "string_decoder",
    "sys",
    "timers",
    "tls",
    "trace_events",
    "tty",
    "url",
    "util",
    "v8",
    "vm",
    "wasi",
    "worker_threads",
    "zlib",
];

/// Whether a specifier names a Node built-in, addressed either with the
/// unambiguous `node:` prefix or as a bare builtin name.
pub(crate) fn is_node_builtin(specifier: &str) -> bool {
    if let Some(builtin) = specifier.strip_prefix("node:") {
        // `node:test`, `node:fs/promises`, etc. The prefix alone is authoritative.
        return !builtin.is_empty();
    }
    let root = specifier.split('/').next().unwrap_or(specifier);
    NODE_BUILTINS.contains(&root)
}

/// Whether a specifier is external (not bundled): a Node built-in. External
/// imports are left in the output for the runtime to resolve.
///
/// This is a property of the SPECIFIER only, so it cannot decide whether leaving
/// the import external is *acceptable*: on a browser target there is no runtime
/// that can resolve it. That decision lives in [`resolve_dependencies`], which
/// knows the [`Target`].
pub(crate) fn is_external_specifier(specifier: &str) -> bool {
    is_node_builtin(specifier)
}

/// The build error for a Node built-in reached from a BROWSER graph. A browser
/// has no `fs`/`net`/`async_hooks`; leaving the import external would emit a
/// chunk whose `require` hits the throw-on-use stub and kills the page at
/// runtime, with a zero exit code at build time. Naming the importer is the
/// whole point: the fix is almost always to stop pulling a server module into
/// the client graph.
fn node_builtin_in_browser_message(path: &Path, specifier: &str) -> String {
    let mut message = format!(
        "Node built-in {specifier:?} cannot be bundled for the browser: browsers have no such module"
    );
    message.push_str(&format!("\n  imported by {}", path.display()));
    message.push_str(
        "\n  a browser build has no Node runtime to resolve it, so this import cannot work at \
         runtime\n  diffpack does NOT implement webpack/Next-style browser polyfills for Node \
         built-ins; that is an unimplemented feature, not a resolution failure\n  either keep this \
         module out of the client graph (import it only from server code), or replace it with a \
         browser-safe equivalent",
    );
    message
}

/// The browser-ESM `requireNative` binding.
///
/// A browser has no `node:module`/`createRequire`, so a `require(...)` that the
/// bundle has no map entry for lands here. There are two genuinely different
/// cases and they must not be conflated:
///
/// - A **Node built-in**. Statically-known built-ins are now a *build* error
///   (see [`node_builtin_in_browser_message`]), so reaching one here means the
///   specifier was only known at runtime. It is bound to a load-safe
///   throw-on-USE stub: property reads and construction succeed (so a module
///   that merely reads a shape off it at init still LOADS), but any actual CALL
///   throws a clear, specifically-named error. It never fabricates a value.
/// - **Anything else** — a package the bundle does not contain, typically an
///   optional dependency required through a specifier the bundler could not see
///   (`require("@emotion/is-prop-" + "valid")`). Node and every other bundler
///   throw *immediately* for that, which is exactly what the near-universal
///   `try { require(optional) } catch {}` idiom is written against. Returning a
///   lazy stub here defeats the `catch`, smuggles the stub into the app as a
///   real value, and blows up much later somewhere unrelated. So: throw now.
///
/// Calling the second case "node builtin ..." — as this stub used to — is simply
/// a false statement about the user's dependency.
fn browser_require_native() -> String {
    let builtins = NODE_BUILTINS
        .iter()
        .map(|name| format!("\"{name}\""))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        r#"const __nodeBuiltins=new Set([{builtins}]);const requireNative=specifier=>{{const builtin=specifier.startsWith("node:")?specifier.length>5:__nodeBuiltins.has(specifier.split("/")[0]);if(!builtin)throw new Error("Cannot require "+JSON.stringify(specifier)+" in the browser: it is not a Node built-in and was not included in the bundle (its specifier is only known at runtime)");const fail=()=>{{throw new Error("node builtin "+specifier+" is not available in the browser");}};const absent=p=>p==="then"||p===Symbol.toPrimitive||p===Symbol.iterator||p===Symbol.asyncIterator;const stub=new Proxy(function(){{fail();}},{{get:(_,p)=>absent(p)?undefined:stub,getOwnPropertyDescriptor:(target,p)=>Reflect.getOwnPropertyDescriptor(target,p)??(typeof p==="string"&&!absent(p)?{{value:stub,writable:true,enumerable:false,configurable:true}}:undefined),has:(target,p)=>absent(p)?Reflect.has(target,p):true,construct:()=>stub,apply:()=>fail()}});return stub;}};"#
    )
}


/// Resolves each transform-detected worker specifier relative to its importer.
/// An unresolvable worker entry is a hard error — the emitted page would 404
/// on `new Worker(...)` at runtime otherwise.
fn resolve_worker_entries(
    resolver: &Resolver,
    importer: &Path,
    workers: &[(String, String)],
) -> Result<Vec<(String, PathBuf)>, String> {
    workers
        .iter()
        .map(|(key, specifier)| {
            resolver
                .resolve_file(importer, specifier)
                .map(|resolution| (key.clone(), resolution.full_path().to_path_buf()))
                .map_err(|error| {
                    format!(
                        "cannot resolve worker entry {specifier:?} from {}: {error}",
                        importer.display()
                    )
                })
        })
        .collect()
}

fn resolve_dependencies(
    resolvers: &Resolvers,
    resolution_cache: &ResolutionCache,
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
            recorded_demand.cloned().unwrap_or_else(|| DependencyDemand {
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
        let also_resolve_as_common_js = recorded_demand
            .is_some_and(|demand| demand.require_syntax && demand.import_syntax);
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
        if is_external_specifier(specifier) {
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
                message: unresolved_import_message(path, specifier, &error.to_string()),
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

/// The package a bare specifier belongs to (`foo/sub` -> `foo`,
/// `@scope/foo/sub` -> `@scope/foo`), or `None` for a relative/absolute path or
/// a `#`-prefixed subpath import, which no `npm install` can fix.
fn bare_package_name(specifier: &str) -> Option<String> {
    if specifier.starts_with('.') || specifier.starts_with('/') || specifier.starts_with('#') {
        return None;
    }
    let mut segments = specifier.split('/');
    let first = segments.next().filter(|segment| !segment.is_empty())?;
    match first.strip_prefix('@') {
        Some(scope) if !scope.is_empty() => {
            let second = segments.next().filter(|segment| !segment.is_empty())?;
            Some(format!("{first}/{second}"))
        }
        _ => Some(first.to_string()),
    }
}

/// URL schemes that address a RESOURCE rather than name a host module. A failure to
/// resolve one of these is a genuine failure — diffpack does not fetch over the
/// network, inline `data:` modules, or accept `blob:` — so they must keep erroring.
const RESOURCE_SCHEMES: &[&str] = &["http", "https", "data", "file", "blob"];

/// The host-runtime scheme of a specifier like `cloudflare:sockets` (`"cloudflare"`),
/// or `None` when the specifier is an ordinary path/package or a resource URL.
///
/// ES modules reserve `scheme:rest` specifiers for host-defined imports, and every
/// JS runtime uses the form for its own built-ins: `node:fs`, `bun:sqlite`,
/// `cloudflare:sockets`, `workerd:*`. They share the properties that matter here —
/// no filesystem lookup can find them and no package manager can install them.
///
/// Deliberately narrow so ordinary specifiers cannot fall in:
/// * the scheme is at least two characters, which excludes a Windows drive (`C:/x`);
/// * the remainder is non-empty and does not start with `/`, which excludes every
///   URL authority form (`https://…`, `file:///…`);
/// * the [`RESOURCE_SCHEMES`] are excluded by name.
///
/// diffpack's own virtual ids (`tanstack-start-manifest:v`) also match the shape, but
/// never reach a caller of this function: they are answered by the virtual-module
/// table at the top of `DirectoryResolutionCache::resolve`, so they resolve
/// successfully and no resolution failure is ever reported for them.
fn host_provided_scheme(specifier: &str) -> Option<&str> {
    let (scheme, rest) = specifier.split_once(':')?;
    if scheme.len() < 2 || rest.is_empty() || rest.starts_with('/') {
        return None;
    }
    if !scheme.chars().next()?.is_ascii_alphabetic()
        || !scheme
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '+' | '-' | '.'))
    {
        return None;
    }
    (!RESOURCE_SCHEMES.contains(&scheme)).then_some(scheme)
}

/// Renders a host-provided module left external on a server graph.
fn host_provided_module_message(path: &Path, specifier: &str) -> String {
    let scheme = host_provided_scheme(specifier).unwrap_or_default();
    let mut message = format!(
        "{specifier:?} names a module the {scheme:?} runtime provides, so it was left external \
         rather than bundled"
    );
    message.push_str(&format!("\n  imported by {}", path.display()));
    message.push_str(
        "\n  a `scheme:` specifier is a host-defined import (like `node:fs`): nothing on disk \
         can satisfy it and no package manager can install it\n  running on that host, the host \
         resolves it; running anywhere else, the import throws exactly where it would without a \
         bundler",
    );
    message
}

/// Renders a specifier one module reaches through both `require(...)` and an ESM
/// `import` where the two syntaxes resolve to different files. Names both files,
/// because the fix is always to pick one syntax and the reader has to see what
/// each one currently gets.
fn specifier_resolves_two_ways_message(
    path: &Path,
    specifier: &str,
    as_import: &Path,
    as_require: &Path,
) -> String {
    let mut message = format!(
        "{} reaches {specifier:?} through both a CommonJS `require(...)` and an ESM `import`, \
         and the two resolve to different files",
        path.display()
    );
    message.push_str(&format!("\n  `import {specifier:?}`  -> {}", as_import.display()));
    message.push_str(&format!("\n  `require({specifier:?})` -> {}", as_require.display()));
    message.push_str(
        "\n  that package's `exports` map sends the two conditions to different builds, so Node \
         itself loads two separate module instances here\n  the bundle records one target per \
         specifier per module, so it cannot carry both, and picking either one would give the \
         other call site the wrong module\n  use ONE syntax for this specifier in this file (or \
         import the exact subpath each call site wants)",
    );
    message
}

/// Renders a missing OPTIONAL dependency. Not an error: it states what was left out
/// and why that is the program's own handled path, so a reader who *did* want the
/// accelerator knows the install that turns it on.
fn optional_dependency_missing_message(path: &Path, specifier: &str) -> String {
    let mut message =
        format!("optional dependency {specifier:?} is not installed, so it was not bundled");
    message.push_str(&format!("\n  required by {}", path.display()));
    message.push_str(
        "\n  every reference to it is a `require(...)` inside a `try` block, so this module \
         already handles it being absent\n  the emitted require throws at exactly the point \
         Node's would, and that module's `catch` runs",
    );
    if let Some(package) = bare_package_name(specifier) {
        message.push_str(&format!("\n  install it to use it:  npm install {package}"));
    }
    message
}

/// Renders an unresolved import: the specifier, the file that imported it, and
/// the action that fixes it. Adapter-generated importers are called out, because
/// pointing a user at a path they never wrote (or, for a virtual module, one
/// that does not exist on disk at all) is the confusing part of this failure.
fn unresolved_import_message(path: &Path, specifier: &str, error: &str) -> String {
    let file_name = path.file_name().and_then(|name| name.to_str());
    let generated = file_name == Some("__diffpack_virtual_module__.js")
        || path.components().any(|component| {
            matches!(
                component.as_os_str().to_str(),
                Some(crate::next_adapter::ADAPTER_DIR) | Some(crate::next_pages::ADAPTER_DIR)
            )
        });
    let mut message = format!("cannot resolve {specifier:?}: {error}");
    if file_name == Some("__diffpack_virtual_module__.js") {
        message.push_str("\n  imported by a diffpack build-generated virtual module");
    } else {
        message.push_str(&format!("\n  imported by {}", path.display()));
        if generated {
            message.push_str("\n              (generated by diffpack, not by your app)");
        }
    }
    // The remedy depends on the specifier's shape: a missing package is an install,
    // a missing file is a typo, and a `#` subpath is a `package.json` `imports` map.
    match bare_package_name(specifier) {
        // The RSC runtime is DIFFPACK's requirement, not the app's: no real Next app
        // depends on `react-server-dom-webpack` (Next vendors its own copy, which
        // diffpack normally resolves — see `rsc_runtime_resolve`). Reaching here means
        // neither the app nor the installed `next` has one, so say whose requirement
        // this is instead of billing the user for a dependency they never declared.
        Some(package) if package == crate::rsc_runtime_resolve::PACKAGE => {
            message.push_str(
                "\n  this is diffpack's requirement, not your app's: diffpack's app-router \
                 entries need an RSC (flight) runtime.\n  It normally uses the copy `next` \
                 vendors at next/dist/compiled/react-server-dom-webpack; the installed `next` \
                 has none (or `next` is not installed).\n  install it:  npm install \
                 react-server-dom-webpack",
            );
        }
        Some(package) => message.push_str(&format!("\n  install it:  npm install {package}")),
        None if specifier.starts_with('#') => message.push_str(
            "\n  a `#` specifier resolves through the nearest package.json `imports` \
             field; check that it maps this specifier",
        ),
        None => message.push_str(
            "\n  no file matched that path; check the spelling and the file extension",
        ),
    }
    message
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
        &source_file,
        target,
        &special.dependency_specifiers,
        &special.dependency_demands,
        diagnostics,
    )
}

#[derive(Clone)]
struct RenderedBundle {
    code: String,
    mappings: Vec<ModuleMapping>,
    /// A fully-composed source-map JSON for this chunk, populated ONLY when the
    /// chunk was minified WITH source maps: it is the composition of the
    /// readable-generated -> original mappings (via [`ModuleMapping`]) through the
    /// minified -> readable-generated map Oxc codegen emits, so a position in the
    /// minified bytes resolves back to the correct ORIGINAL source file+region.
    /// When `None`, [`Self::mappings`] describes the emitted bytes directly (the
    /// readable, un-minified output) and the map is built from them at write time.
    map_json: Option<String>,
}

#[derive(Clone)]
struct ModuleMapping {
    dense_index: DenseModuleId,
    generated_line: u32,
    /// The module's REAL map tokens, already moved onto the chunk's generated
    /// lines: `generated_line`/`generated_column` are positions in the CHUNK,
    /// `source_line`/`source_column` positions in the module's source, and
    /// `name` an index into that module's map names. Empty when the build did
    /// not ask for source maps, and empty for the parts of a region whose text
    /// the render rewrote in a way it could not account for — those stay
    /// UNMAPPED. See [`crate::source_map`].
    tokens: Vec<MapToken>,
}

/// A per-chunk render cache, keyed by a stable [`Bundler::chunk_render_key`]: the
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
#[derive(Default)]
struct RenderCache {
    entries: HashMap<u64, RenderedBundle>,
}

/// What a single [`Bundler::emit_with_options`] wrote and re-rendered. The
/// `rendered_chunks` count is the incrementality signal (a leaf edit re-renders
/// exactly one chunk); `written` is the set of files kept on disk, so the
/// environment emit can delete only files that are no longer part of the build
/// instead of nuking the whole output tree.
#[derive(Debug, Default)]
pub struct EmitStats {
    pub rendered_chunks: usize,
    written: BTreeSet<PathBuf>,
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
#[derive(Copy, Clone)]
pub struct EmitCancel<'a>(Option<&'a (dyn Fn() -> bool + Send + Sync)>);

impl<'a> EmitCancel<'a> {
    /// An emit that always runs to completion. Every production build path.
    pub fn never() -> Self {
        Self(None)
    }

    /// Stop when `signal` returns true. It is called often, from rayon worker
    /// threads, so it must be cheap (an atomic load) and must not block.
    pub fn when(signal: &'a (dyn Fn() -> bool + Send + Sync)) -> Self {
        Self(Some(signal))
    }

    /// Whether the work should stop. Called from the emit's own phases and from the
    /// Tailwind delegate, which polls it while the app's compiler runs.
    pub fn cancelled(&self) -> bool {
        self.0.is_some_and(|signal| signal())
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

/// Full minification of one FINISHED chunk's JavaScript.
///
/// The chunk `code` handed in is already clean, valid JS (the marker-based linker
/// consumed its markers during render; it passes `node --check` and runs
/// in-browser), so this is a self-contained final pass: re-parse the emitted bytes,
/// run `oxc_minifier` (compression + identifier mangling) over the program, and
/// re-print it with Oxc codegen configured for minified output. It never touches
/// the linker.
///
/// Running the minifier per FINISHED chunk rather than over the module graph is
/// what keeps it compatible with the incremental thesis: a chunk whose bytes did
/// not change is not re-minified at all (the render cache is keyed upstream of
/// this), and one chunk's compression can never depend on another's contents.
///
/// Mangling is safe across the module registry because every cross-module name
/// crosses the boundary as a STRING, not an identifier: exports are published with
/// `__export(exports, "name", () => local)` and imports resolve through
/// `__maps[id]["specifier"]`. The mangler renames `local`; the string keys it is
/// looked up by are literals and are untouched.
///
/// A parse failure on the generated chunk is a HARD error naming the chunk, never
/// a silent passthrough of the unminified bytes.
fn minify_chunk_code(code: &str, chunk_name: &str) -> Result<String, String> {
    Ok(minify_chunk_code_inner(code, chunk_name, false)?.0)
}

/// Like [`minify_chunk_code`], but also returns the Oxc codegen source map from
/// the MINIFIED bytes back to the readable-generated `code` it was handed. That
/// map is later composed (`Bundler::compose_source_map`) with the readable ->
/// original module mappings so a minified position resolves to the correct
/// original source. Oxc returning no map despite source-map output being
/// requested is a hard error naming the chunk, never a silently mapless minify.
fn minify_chunk_code_with_map(
    code: &str,
    chunk_name: &str,
) -> Result<(String, oxc_sourcemap::SourceMap<'static>), String> {
    let (minified, map) = minify_chunk_code_inner(code, chunk_name, true)?;
    let map = map.ok_or_else(|| {
        format!(
            "minify: Oxc codegen returned no source map for chunk `{chunk_name}` despite \
             source-map output being requested"
        )
    })?;
    Ok((minified, map))
}

/// The shared minify pass: re-parse the finished readable chunk and re-print it
/// minified, optionally producing the minified -> readable source map. The map is
/// converted to `'static` (`into_owned`) so it outlives the parse allocator.
fn minify_chunk_code_inner(
    code: &str,
    chunk_name: &str,
    want_map: bool,
) -> Result<(String, Option<oxc_sourcemap::SourceMap<'static>>), String> {
    use oxc_allocator::Allocator;
    use oxc_codegen::{Codegen, CodegenOptions};
    use oxc_minifier::{Minifier, MinifierOptions};
    use oxc_parser::Parser;
    use oxc_span::SourceType;

    let allocator = Allocator::default();
    // Every emitted chunk (browser ESM entry/chunks and Node `.mjs`) is module
    // JavaScript; parse it as such so top-level `import`/`export` are accepted.
    let source_type = SourceType::default().with_module(true);
    let parsed = Parser::new(&allocator, code, source_type).parse();
    if parsed.panicked || !parsed.diagnostics.is_empty() {
        let detail = parsed
            .diagnostics
            .first()
            .map(|error| error.to_string())
            .unwrap_or_else(|| "parser panicked".to_string());
        return Err(format!(
            "minify: cannot parse generated chunk `{chunk_name}` for minification: {detail}"
        ));
    }
    let mut program = parsed.program;
    // Compression + mangling. The defaults match what esbuild (and therefore Vite)
    // applies to a production build, which is the comparison this output is held
    // to.
    let minified = Minifier::new(MinifierOptions::default()).minify(&allocator, &mut program);
    // `CodegenOptions::minify()` already drops comments and collapses whitespace;
    // set `source_map_path` only when a map is wanted (it enables the codegen map,
    // whose `source` is this chunk's readable bytes — the composition re-attaches
    // the real original sources, so the exact path here is immaterial).
    let mut options = CodegenOptions::minify();
    if want_map {
        options.source_map_path = Some(PathBuf::from(chunk_name));
    }
    let mut codegen = Codegen::new().with_options(options);
    // The mangler records its renames in the returned scoping; codegen must print
    // through it or the mangled bindings and their references disagree.
    if let Some(scoping) = minified.scoping {
        codegen = codegen.with_scoping(Some(scoping));
    }
    let printed = codegen.build(&program);
    let map = printed.map.map(|map| map.into_owned());
    Ok((printed.code, map))
}

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
struct ResolvedMinifiedToken<'a> {
    dense: DenseModuleId,
    source_line: u32,
    source_column: u32,
    name: Option<&'a str>,
}

/// Exactly `readable.partition_point(|(token, _)| (line, column) <= position)`,
/// found by expanding exponentially outward from `hint` and binary-searching the
/// bracketed range. For the nearly-sorted queries source-map composition makes
/// (each token's answer sits within a few entries of the previous one's) this
/// costs O(log distance-from-hint) local probes instead of O(log n) random ones.
fn partition_point_from_hint(
    readable: &[(MapToken, DenseModuleId)],
    position: (u32, u32),
    hint: usize,
) -> usize {
    let at_or_before = |index: usize| {
        let token = &readable[index].0;
        (token.generated_line, token.generated_column) <= position
    };
    let length = readable.len();
    if length == 0 {
        return 0;
    }
    let anchor = hint.min(length - 1);
    let (mut low, mut high);
    if at_or_before(anchor) {
        // The partition point is right of `anchor`.
        low = anchor + 1;
        high = length;
        let mut width = 1;
        while anchor + width < length {
            let probe = anchor + width;
            if at_or_before(probe) {
                low = probe + 1;
                width *= 2;
            } else {
                high = probe;
                break;
            }
        }
    } else {
        // The partition point is at or left of `anchor`.
        low = 0;
        high = anchor;
        let mut width = 1;
        while width <= anchor {
            let probe = anchor - width;
            if at_or_before(probe) {
                low = probe + 1;
                break;
            }
            high = probe;
            width *= 2;
        }
    }
    while low < high {
        let middle = low + (high - low) / 2;
        if at_or_before(middle) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low
}

fn is_identifier(name: &str) -> bool {
    let mut characters = name.chars();
    characters
        .next()
        .is_some_and(|first| first.is_alphabetic() || first == '_' || first == '$')
        && characters.all(|character| character.is_alphanumeric() || character == '_' || character == '$')
}

/// How many lines `text` has, in the numbering a source map uses: the last line
/// counts even when the text does not end with a newline, and a trailing newline
/// does not invent an extra one. Generated line indices run over exactly this
/// range, so it is what bounds the unmapped markers a chunk map emits.
fn line_count(text: &str) -> u32 {
    text.lines().count() as u32
}

/// Byte offset of the start of each line of `text`.
fn line_starts(text: &str) -> Vec<usize> {
    let mut starts = vec![0];
    starts.extend(
        text.match_indices('\n')
            .map(|(index, _)| index + 1),
    );
    starts
}

/// The JavaScript identifier that STARTS at `line`:`column` of `text`, where the
/// column is in UTF-16 code units (the source-map unit). `None` when the position
/// is out of range or does not begin an identifier — which is the answer that
/// keeps a name out of the map rather than publishing one the source does not have.
fn identifier_at<'a>(text: &'a str, starts: &[usize], line: u32, column: u32) -> Option<&'a str> {
    let start = *starts.get(line as usize)?;
    let end = starts
        .get(line as usize + 1)
        .map_or(text.len(), |next| next - 1);
    let line_text = text.get(start..end)?;
    let mut offset = 0;
    let mut units = 0_u32;
    for character in line_text.chars() {
        if units == column {
            break;
        }
        units += character.len_utf16() as u32;
        offset += character.len_utf8();
    }
    if units != column {
        return None;
    }
    let rest = &line_text[offset..];
    let is_start = |character: char| character.is_ascii_alphabetic() || character == '_' || character == '$';
    let is_part = |character: char| character.is_ascii_alphanumeric() || character == '_' || character == '$';
    if !rest.starts_with(is_start) {
        return None;
    }
    let length = rest.find(|character: char| !is_part(character)).unwrap_or(rest.len());
    Some(&rest[..length])
}

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

/// The chunk files a browser must evaluate, in order, to have `index` fully
/// loaded: the transitive prerequisite closure first (post-order, so a chunk never
/// precedes something it depends on), then the chunk itself. Used for the route
/// manifest's preload lists, where the order IS the contract.
fn chunk_load_order(
    plans: &[ChunkPlan],
    index: usize,
    seen: &mut HashSet<usize>,
    ordered: &mut Vec<String>,
) {
    if !seen.insert(index) {
        return;
    }
    for &prerequisite in &plans[index].prerequisites {
        chunk_load_order(plans, prerequisite, seen, ordered);
    }
    ordered.push(plans[index].file_name.clone());
}

/// The emitted file name for the chunk that owns dynamic root `index` (1-based,
/// its position in [`Bundler::dynamic_roots`]) and nothing else. Most roots use
/// the numbered `<stem>.chunk-<index>` name; the build-generated
/// `tanstack-start-manifest:v` virtual module keeps a descriptive
/// `_tanstack-start-manifest_v` name so the emitted artifact is identifiable (and
/// matches TanStack's own manifest chunk naming convention).
fn chunk_file_name(stem: &str, extension: &str, index: usize, id: &str) -> String {
    if id == crate::manifest::START_MANIFEST_SPECIFIER {
        return format!("_tanstack-start-manifest_v{extension}");
    }
    format!("{stem}.chunk-{index}{extension}")
}

/// Splits an entry file name (`client.js`) into its stem (`client`) and
/// dotted extension (`.js`), the two halves every chunk name in
/// [`Bundler::chunk_plan`] is built from.
fn split_file_name(file: &str) -> Result<(String, String), String> {
    let path = Path::new(file);
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| format!("entry file has no stem: {file}"))?;
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map_or(String::new(), |extension| format!(".{extension}"));
    Ok((stem.to_string(), extension))
}

/// The TanStack route id a dynamic chunk belongs to, when the chunk is a route's
/// `?tsr-split=*` split module. Returns `Ok(None)` for a non-route-split chunk
/// (which is a real chunk but not a route preload). A route-split chunk whose
/// route id cannot be derived is a hard error, never a silently dropped preload.
fn split_chunk_route_id(id: &str) -> Result<Option<String>, String> {
    let resource = ResourceId::parse(id);
    if resource.loader_kind() != Some(LoaderKind::TsrSplit) {
        return Ok(None);
    }
    let path = Path::new(&resource.path);
    let source = fs::read_to_string(path)
        .map_err(|error| format!("cannot read route file {}: {error}", path.display()))?;
    match crate::route_split::route_id(path, &source) {
        Some(route_id) => Ok(Some(route_id)),
        None => Err(format!(
            "route split chunk {id} has no derivable TanStack route id \
             (the createFileRoute string argument); cannot attribute its preload to a route"
        )),
    }
}

/// How a default asset import of a raster image (`import img from './x.png'`)
/// materializes. `Url` (the default, and what Vite/TanStack/generic builds use)
/// makes the default export the bare public URL string, byte-identical to Vite.
/// `NextObject` makes it Next's static-import object shape
/// (`{ src, width, height, blurDataURL, variants }`) with build-emitted responsive
/// variants — set ONLY by the Next app-router adapter so no other build path
/// changes its asset-import semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageImportShape {
    /// Bare public URL string (Vite parity). The default.
    #[default]
    Url,
    /// Next's static-import object shape with blur, and — when
    /// `responsive_variants` is set — a build-emitted responsive ladder.
    ///
    /// `responsive_variants` is false when the app's next.config turned Next's image
    /// optimizer off (`images.unoptimized`) or replaced it with its own loader: the
    /// object then carries `src`/`width`/`height`/`blurDataURL` but NO `variants`, so
    /// the `next/image` shim renders a plain `<img src>` with no `srcset` — exactly
    /// what `next build` produces, and with no ladder encoded for URLs that can never
    /// be requested.
    NextObject { responsive_variants: bool },
}

/// CSS preprocessor + PostCSS wiring for a build. `.less`/`.styl` sources are
/// compiled by the app's own Less/Stylus (`node`, cwd = `root`); a discovered
/// PostCSS config runs the app's own plugins over every stylesheet. Default is
/// off: no PostCSS, and Less/Stylus resolve their tool from each file's own
/// directory. See [`crate::less_stylus`] and [`crate::postcss`].
#[derive(Debug, Clone, Default)]
pub struct CssPreprocess {
    /// Project root, used as the `node` working directory so the app's own
    /// `less`/`stylus`/`postcss` (and plugins) resolve from its node_modules.
    pub root: Option<PathBuf>,
    /// The discovered PostCSS setup, shared across the build (`None` = no
    /// PostCSS step). Behind an `Arc` because [`BuildConfig`] is `Clone` and the
    /// setup owns an interior content cache.
    pub postcss: Option<Arc<crate::postcss::Postcss>>,
}

impl CssPreprocess {
    fn root_path(&self) -> Option<&Path> {
        self.root.as_deref()
    }
}

/// Build-time configuration a plugin host contributes. Currently the resolver
/// aliases (specifier -> absolute target), such as TanStack's
/// `#tanstack-router-entry` -> `<app>/src/router.tsx`. Kept small and owned by
/// Rust; the host merely supplies the values.
#[derive(Debug, Clone)]
pub struct BuildConfig {
    /// Install the browser `process.env` shim in client output. Vite does NOT
    /// (a Vite browser build has `typeof process === "undefined"`, and
    /// env-sniffing libraries take browser paths because of it — found live:
    /// i18next behaved differently under the shim). Only the TanStack
    /// `build-app` path needs it, for runtime reads of `TSS_SERVER_FN_BASE`
    /// and `NODE_ENV` in vendored code the compile-time define cannot reach.
    pub browser_process_shim: bool,
    /// Assets at or under this many bytes are inlined as `data:` URIs instead
    /// of emitted files (Vite's `assetsInlineLimit`, default 4096 in Vite
    /// mode). `0` disables inlining — the default for generic bundling.
    pub asset_inline_limit: usize,
    /// The public base URL every emitted asset URL is prefixed with. `"/"`
    /// unless the build opts into a Vite config with a non-root `base` (a site
    /// served from a subpath, e.g. GitHub Pages). Always ends with `/`.
    pub base: String,
    /// Ordered `(specifier, absolute_target)` alias pairs.
    pub aliases: Vec<(String, String)>,
    /// Environment resolve conditions (e.g. client `["module","browser",
    /// "production"]`, server `["node",...]`). This is what isolates client from
    /// server: browser conditions select packages' browser exports and exclude
    /// server-only code. Empty means the built-in default.
    pub conditions: Vec<String>,
    /// Vite's `resolve.mainFields`: the `package.json` fields to try, in order,
    /// when a package has no `exports` map. Empty keeps the built-in per-target
    /// default (`["browser","module","main"]` for the client, `["module","main"]`
    /// for the server).
    pub main_fields: Vec<String>,
    /// Build-generated virtual modules, `(specifier, module_source)`. A specifier
    /// listed here resolves to itself (a virtual id) and loads from the given
    /// source instead of the filesystem. Used for the natively generated
    /// `tanstack-start-manifest:v` module, whose contents depend on the client
    /// build's chunk graph and so cannot be read from a package.
    pub virtual_modules: Vec<(String, String)>,
    /// The environment being compiled. Selects TanStack Start's per-environment
    /// specialization of directive helpers (see [`Target`]); defaults to the
    /// server (no transform).
    pub target: Target,
    /// Vite's `import.meta.env` values, when the build opts into that convention
    /// (the `build-app` path sets it). `None` for generic bundling, which leaves
    /// `import.meta.env` untouched. See [`crate::import_meta_env`].
    pub import_meta_env: Option<crate::import_meta_env::ImportMetaEnv>,
    /// Vite's `import.meta.glob` expansion, when the build opts into that
    /// convention (set alongside `import_meta_env` by the Vite-convention build
    /// paths). `None` for generic bundling, which leaves `import.meta.glob`
    /// untouched. See [`crate::import_meta_glob`].
    pub import_meta_glob: Option<crate::import_meta_glob::ImportMetaGlob>,
    /// Vite `define` entries as `(identifier, replacement_source)`, evaluated once
    /// from the config. Empty for generic bundling. See [`crate::vite_define`].
    pub defines: Vec<(String, String)>,
    /// DEV-ONLY: instrument client component modules with React Fast Refresh and
    /// rewrite `import.meta.hot`. Set only by the dev server; `build-app` leaves it
    /// `false` so production output is unaffected. See [`crate::hmr`].
    pub hmr: bool,
    /// Produce a REAL per-module source map during the transform. Costs one extra
    /// print per module (the Oxc printer only emits a map for a whole `Program`,
    /// so the map comes from a second, reference print), so it is paid only when
    /// the emit will actually write `.map` files. Correctness is never
    /// conditional on this: when it is off no map is written at all, rather than
    /// a cheaper, guessed one. See [`crate::source_map`].
    pub source_maps: bool,
    /// SCSS compile options: Vite's `css.preprocessorOptions.scss.additionalData`
    /// (when a string) and the project root for root-relative `@use "/src/..."`
    /// targets. Default (empty) compiles `.scss` files with no injected prelude.
    pub scss: crate::sass::ScssOptions,
    /// How default raster-image imports materialize. `Url` (default) keeps Vite
    /// parity (bare URL string); the Next adapter opts into `NextObject` so
    /// `import img from './x.png'` yields Next's `{ src, width, height,
    /// blurDataURL, variants }` shape with build-emitted responsive variants.
    pub image_import_shape: ImageImportShape,
    /// Less/Stylus compilation + PostCSS. Default is off (no PostCSS; Less/Stylus
    /// resolve their tool from each file's directory). See [`CssPreprocess`].
    pub css_preprocess: CssPreprocess,
    /// Which extensions may contain JSX. `JsxAndTsxOnly` (default) is the
    /// Vite/esbuild rule; the Next adapters opt into `NextJs`, where `.js`/`.mjs`/
    /// `.cjs` are JSX-capable too. See [`crate::parser::JsxExtensions`].
    pub jsx_extensions: crate::parser::JsxExtensions,
    /// How JSX is LOWERED, as the BUILD configures it: `vite.config`'s
    /// `esbuild.{jsx,jsxImportSource,jsxFactory,jsxFragment}` / `oxc.jsx`. Layered
    /// over the tsconfig that owns each file (which is honored in every mode,
    /// because a tsconfig is the file's own compilation contract). Default (all
    /// `None`) is the automatic runtime against `react`.
    pub jsx: crate::transform::JsxConfig,
    /// Package names a SERVER graph must NOT bundle: they stay ordinary runtime
    /// `require`s resolved from `node_modules` at serve time. This is Next's
    /// `serverExternalPackages` (formerly
    /// `experimental.serverComponentsExternalPackages`), and honoring it is not an
    /// optimization — apps put a package here precisely BECAUSE bundling it fails.
    /// cal.com externalizes `rest-facade`, whose `require('superagent-proxy')` sits
    /// behind a runtime `if` and names a package that is not installed; bundling it
    /// turns an untaken branch into a fatal unresolved-import error.
    ///
    /// Empty (the default) bundles everything, and a CLIENT build ignores the list —
    /// the browser has no `node_modules` to require from at runtime.
    pub server_external_packages: Vec<String>,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            // The root base, not an empty string: every minted asset URL is
            // `{base}assets/...`, and `/assets/...` is the correct default.
            base: "/".to_string(),
            browser_process_shim: false,
            asset_inline_limit: 0,
            aliases: Vec::new(),
            conditions: Vec::new(),
            main_fields: Vec::new(),
            virtual_modules: Vec::new(),
            target: Target::default(),
            import_meta_env: None,
            import_meta_glob: None,
            defines: Vec::new(),
            hmr: false,
            source_maps: false,
            scss: crate::sass::ScssOptions::default(),
            image_import_shape: ImageImportShape::Url,
            css_preprocess: CssPreprocess::default(),
            jsx_extensions: crate::parser::JsxExtensions::JsxAndTsxOnly,
            jsx: crate::transform::JsxConfig::default(),
            server_external_packages: Vec::new(),
        }
    }
}

/// The two ESM-only export conditions. A `require(...)` call site must not
/// resolve under either: `package.json`'s `exports` is a MAP from condition to
/// file, so leaving `import` in the set for a `require` resolution picks the ESM
/// file whenever the package lists `import` first — which is the whole reason
/// dual-package publishing works at all.
const ESM_ONLY_CONDITIONS: [&str; 2] = ["import", "module"];

/// The resolver options for one dependency SYNTAX. Everything except the export
/// conditions is shared: a `require(...)` and an `import` from the same file walk
/// the same `node_modules`, honour the same tsconfig paths and the same `browser`
/// field. Only which key of a package's `exports` map answers differs.
fn resolve_options_for_syntax(config: &BuildConfig, syntax: ImportSyntax) -> ResolveOptions {
    // Without host-supplied conditions, keep the built-in default. With them (the
    // environment's browser/node conditions), keep `import`/`default` too so basic
    // ESM resolution still works.
    let condition_names = match syntax {
        ImportSyntax::Esm if config.conditions.is_empty() => {
            vec!["import".into(), "module".into(), "default".into()]
        }
        ImportSyntax::Esm => {
            let mut names = config.conditions.clone();
            for fallback in ["import", "default"] {
                if !names.iter().any(|name| name == fallback) {
                    names.push(fallback.to_string());
                }
            }
            names
        }
        // A CommonJS `require(...)`: `require` replaces the ESM-only conditions
        // rather than joining them, because the set is matched against the
        // package's own key ORDER and any surviving `import`/`module` key would
        // win first. Every non-ESM condition the host asked for (`node`,
        // `browser`, `react-server`, `production`) still applies — those describe
        // the ENVIRONMENT, which a `require` does not change.
        ImportSyntax::CommonJs => {
            let mut names = config
                .conditions
                .iter()
                .filter(|name| !ESM_ONLY_CONDITIONS.contains(&name.as_str()))
                .cloned()
                .collect::<Vec<String>>();
            for fallback in ["require", "default"] {
                if !names.iter().any(|name| name == fallback) {
                    names.push(fallback.to_string());
                }
            }
            names
        }
    };
    ResolveOptions {
        tsconfig: Some(TsconfigDiscovery::Auto),
        extensions: [".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".json", ".mdx", ".md"]
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
        // A browser build must honor `package.json`'s `browser` field (the
        // classic pre-exports substitution map: `debug` swaps its Node entry
        // for `src/browser.js`, and packages stub out `fs` etc. with `false`).
        // Without it the Node implementation leaks into the client graph and
        // drags in Node-only optional dependencies. Server builds must NOT
        // apply it — they want the real Node entries.
        alias_fields: if config.target == Target::Client {
            vec![vec!["browser".into()]]
        } else {
            Vec::new()
        },
        // Vite's `resolve.mainFields` overrides the per-target default when set;
        // otherwise keep the built-in default (browser fields for the client).
        main_fields: if !config.main_fields.is_empty() {
            config.main_fields.clone()
        } else if config.target == Target::Client {
            vec!["browser".into(), "module".into(), "main".into()]
        } else {
            vec!["module".into(), "main".into()]
        },
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

pub(crate) fn content_hash(bytes: &[u8]) -> u64 {
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

    /// The hinted partition search must return EXACTLY `partition_point`'s
    /// answer for every (query, hint) pair — it is the thing that makes the
    /// composed source map identical to the single-cursor composition. Checked
    /// exhaustively over a readable array with duplicate positions, line
    /// boundaries, and gaps, for every hint from 0 past the end.
    #[test]
    fn partition_point_from_hint_matches_partition_point_for_every_hint() {
        let token = |line: u32, column: u32| MapToken {
            generated_line: line,
            generated_column: column,
            source_line: 0,
            source_column: 0,
            name: None,
        };
        let readable: Vec<(MapToken, DenseModuleId)> = [
            (0, 0),
            (0, 4),
            (0, 4),
            (0, 9),
            (1, 0),
            (3, 2),
            (3, 2),
            (3, 7),
            (7, 0),
            (7, 1),
        ]
        .into_iter()
        .map(|(line, column)| (token(line, column), 0))
        .collect();
        for query_line in 0..9u32 {
            for query_column in 0..11u32 {
                let position = (query_line, query_column);
                let expected = readable.partition_point(|(token, _)| {
                    (token.generated_line, token.generated_column) <= position
                });
                for hint in 0..=readable.len() + 2 {
                    assert_eq!(
                        partition_point_from_hint(&readable, position, hint),
                        expected,
                        "position {position:?} hint {hint}"
                    );
                }
            }
        }
        assert_eq!(partition_point_from_hint(&[], (5, 5), 3), 0);
    }

    /// `node`, with an inherited terminal-colour override stripped.
    ///
    /// Many tests here execute an emitted chunk and compare its stdout
    /// byte-for-byte. `console.log(6)` prints `6` down a pipe but
    /// `\x1b[33m6\x1b[39m` when node believes it is writing to a terminal, and
    /// an inherited `FORCE_COLOR` (set by plenty of terminal wrappers and CI
    /// runners) makes it believe exactly that. Every such assertion then fails
    /// for a reason that has nothing to do with the bundler. Removing the
    /// variable — rather than setting `NO_COLOR`, which node ignores in its
    /// presence and warns about on stderr — makes the output environment-
    /// independent.
    fn node_command() -> Command {
        let mut command = Command::new("node");
        command.env_remove("FORCE_COLOR");
        command
    }

    #[test]
    fn node_is_spawned_without_inherited_terminal_colour() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        // The hazard is real: with FORCE_COLOR set, node writes ANSI escapes even
        // down a pipe, and every stdout comparison in this module would fail.
        let coloured = Command::new("node")
            .env("FORCE_COLOR", "3")
            .args(["-e", "console.log(6)"])
            .output()
            .unwrap();
        assert!(
            String::from_utf8_lossy(&coloured.stdout).contains('\u{1b}'),
            "expected node to colour its output under FORCE_COLOR"
        );
        // node_command() unsets it, whatever the parent environment holds.
        assert!(
            node_command()
                .get_envs()
                .any(|(key, value)| key == std::ffi::OsStr::new("FORCE_COLOR") && value.is_none()),
            "node_command must remove FORCE_COLOR from the child environment"
        );
        let plain = node_command().args(["-e", "console.log(6)"]).output().unwrap();
        assert_eq!(String::from_utf8_lossy(&plain.stdout), "6\n");
    }

    #[test]
    fn bundles_typescript_dynamic_import_and_a_package_into_executable_javascript() {
        if node_command().arg("--version").output().is_err() {
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

        let executed = node_command().arg(&output).output().unwrap();
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

    /// A dual-published package: `exports` sends `import` and `require` to two
    /// different files, exactly as `pg-pool` (and most of npm) does.
    fn write_dual_package(directory: &Path, name: &str, esm_body: &str, cjs_body: &str) {
        let package = directory.join("node_modules").join(name);
        fs::create_dir_all(&package).unwrap();
        fs::write(
            package.join("package.json"),
            format!(
                r#"{{"name":"{name}","exports":{{".":{{"import":"./esm.mjs","require":"./cjs.js"}}}}}}"#
            ),
        )
        .unwrap();
        fs::write(package.join("esm.mjs"), esm_body).unwrap();
        fs::write(package.join("cjs.js"), cjs_body).unwrap();
    }

    /// A `require(...)` call site must resolve under the `require` export
    /// condition. Resolving it under `import` hands back a Module namespace where
    /// the caller expects the CommonJS export, and `class extends <namespace>`
    /// throws `Class extends value [object Module] is not a constructor` — which
    /// is exactly how `pg`'s `require('pg-pool')` died.
    #[test]
    fn a_require_call_site_resolves_under_the_require_condition() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        write_dual_package(
            directory.path(),
            "dual",
            "export default class Esm {}\n",
            "class Cjs {}\nmodule.exports = Cjs;\n",
        );
        fs::write(
            directory.path().join("entry.js"),
            "const Base = require(\"dual\");\nclass Sub extends Base {}\nconsole.log(new Sub().constructor.name === \"Sub\" ? Base.name : \"wrong\");\n",
        )
        .unwrap();
        assert_eq!(bundle_and_run(directory.path()), "Cjs\n");
    }

    /// The other half of the same rule: an `import` of the identical specifier
    /// still resolves under `import`.
    #[test]
    fn an_import_of_the_same_package_still_resolves_under_the_import_condition() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        write_dual_package(
            directory.path(),
            "dual",
            "export const which = \"esm\";\n",
            "exports.which = \"cjs\";\n",
        );
        fs::write(
            directory.path().join("entry.js"),
            "import { which } from \"dual\";\nconsole.log(which);\n",
        )
        .unwrap();
        assert_eq!(bundle_and_run(directory.path()), "esm\n");
    }

    /// One module, one specifier, both syntaxes, two different files. Node loads
    /// two module instances here and the runtime map holds one target per
    /// specifier, so the build refuses rather than silently giving one call site
    /// the other's module.
    #[test]
    fn one_specifier_reached_both_ways_that_resolves_two_ways_is_a_hard_error() {
        let directory = tempdir().unwrap();
        write_dual_package(
            directory.path(),
            "dual",
            "export const which = \"esm\";\n",
            "exports.which = \"cjs\";\n",
        );
        fs::write(
            directory.path().join("entry.js"),
            "const eager = require(\"dual\");\nexport const lazy = import(\"dual\");\nconsole.log(eager, lazy);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (_, update) = Bundler::discover(&entry).unwrap();
        let fatal = update
            .diagnostics
            .iter()
            .find(|diagnostic| {
                matches!(diagnostic.kind, DiagnosticKind::SpecifierResolvesTwoWays { .. })
            })
            .expect("reaching one specifier both ways must be reported");
        assert!(fatal.is_fatal());
        assert!(fatal.message.contains("entry.js"), "{}", fatal.message);
        assert!(fatal.message.contains("esm.mjs"), "{}", fatal.message);
        assert!(fatal.message.contains("cjs.js"), "{}", fatal.message);
    }

    /// A package whose `exports` sends both conditions to the SAME file is not a
    /// conflict, so reaching it both ways is fine.
    #[test]
    fn one_specifier_reached_both_ways_that_resolves_the_same_way_is_not_an_error() {
        let directory = tempdir().unwrap();
        let package = directory.path().join("node_modules/single");
        fs::create_dir_all(&package).unwrap();
        fs::write(
            package.join("package.json"),
            r#"{"name":"single","exports":{".":{"import":"./index.js","require":"./index.js"}}}"#,
        )
        .unwrap();
        fs::write(package.join("index.js"), "exports.which = \"one\";\n").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "const eager = require(\"single\");\nexport const lazy = import(\"single\");\nconsole.log(eager, lazy);\n",
        )
        .unwrap();
        let (_, update) = Bundler::discover(&directory.path().join("entry.js")).unwrap();
        assert!(
            !update.diagnostics.iter().any(|diagnostic| matches!(
                diagnostic.kind,
                DiagnosticKind::SpecifierResolvesTwoWays { .. }
            )),
            "{:?}",
            update.diagnostics
        );
    }

    /// `export const p = import("./a")` holds a real dependency. The dependency
    /// scan used to stop at the `from` clause of an `export … from` and never
    /// look inside an exported declaration, so this module was bundled with no
    /// edge at all and the emitted `import()` threw MODULE_NOT_FOUND.
    #[test]
    fn a_dynamic_import_inside_an_exported_declaration_is_bundled_and_runs() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("lazy.js"),
            "export const value = \"lazy-value\";\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("map.js"),
            "export const Map = { a: import(\"./lazy.js\") };\nexport const run = async () => (await Map.a).value;\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { run } from \"./map.js\";\nrun().then((v) => console.log(v));\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = Bundler::discover(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        assert_eq!(reachable.len(), 3, "the dynamic target must be in the graph");
        assert_eq!(bundle_and_run(directory.path()), "lazy-value\n");
    }

    /// A `require(...)` inside an exported declaration is the same hole.
    #[test]
    fn a_require_inside_an_exported_declaration_is_bundled_and_runs() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("dep.js"), "exports.value = \"dep-value\";\n").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "export const dep = require(\"./dep.js\");\nconsole.log(dep.value);\n",
        )
        .unwrap();
        assert_eq!(bundle_and_run(directory.path()), "dep-value\n");
    }

    /// Bundles `entry.js` out of an already-populated directory and runs the
    /// result under Node, returning its stdout. The interop tests below are all
    /// "what does the emitted program actually print", which is the only level
    /// at which a runtime helper can be pinned.
    fn bundle_and_run(directory: &Path) -> String {
        let entry = directory.join("entry.js");
        let output = directory.join("dist/bundle.js");
        let (bundler, update) = Bundler::discover(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();
        let executed = node_command().arg(&output).output().unwrap();
        assert!(
            executed.status.success(),
            "{}",
            String::from_utf8_lossy(&executed.stderr)
        );
        String::from_utf8_lossy(&executed.stdout).into_owned()
    }

    /// `import { missing } from "./legacy.cjs"` is a hard error in Node, and
    /// must not evaluate to `undefined` here either — not even when the module
    /// stamps the `__esModule` convention marker on itself, which is exactly
    /// the case the interop's own CommonJS marker used to wave through.
    #[test]
    fn a_named_import_a_commonjs_module_does_not_provide_is_a_hard_error() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("marked.cjs"),
            "exports.__esModule = true;\nexports.present = \"present-val\";\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            r#"
                import { present, missingName } from "./marked.cjs";
                import marked from "./marked.cjs";
                console.log("present:" + present);
                console.log("default-is-module-exports:" + (marked.present === "present-val"));
                try {
                  console.log("missing:" + missingName);
                } catch (error) {
                  console.log("threw:" + (error instanceof SyntaxError));
                  console.log("message:" + error.message);
                }
            "#,
        )
        .unwrap();

        let stdout = bundle_and_run(directory.path());
        let lines = stdout.lines().collect::<Vec<_>>();
        assert_eq!(
            &lines[..3],
            [
                "present:present-val",
                "default-is-module-exports:true",
                "threw:true",
            ],
            "{stdout}"
        );
        // The error names both the module and the export, the way Node's does.
        let message = lines[3];
        assert!(message.contains("./marked.cjs"), "{message}");
        assert!(message.contains("missingName"), "{message}");
    }

    /// The `__esModule` interop. A CommonJS module that stamps the marker AND owns a
    /// `default` was compiled down from ESM (TypeScript / Babel / SWC output, which is
    /// most of npm), so `import X from` it must bind THAT default — not the exports
    /// object wrapping it. Binding the wrapper is silent until the value is used as
    /// what it claims to be: `next-auth/providers/credentials` is exactly this shape,
    /// and cal.com's next-auth config died on `o(...) is not a function` because the
    /// provider factory came back as `{ __esModule: true, default: fn }`.
    #[test]
    fn a_default_import_of_a_transpiled_commonjs_module_binds_its_default_export() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("provider.cjs"),
            "Object.defineProperty(exports, \"__esModule\", { value: true });\n\
             exports.default = Credentials;\n\
             exports.named = \"named-val\";\n\
             function Credentials(options) { return { id: \"credentials\", options }; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            r#"
                import Credentials from "./provider.cjs";
                import { named } from "./provider.cjs";
                import * as ns from "./provider.cjs";
                console.log("typeof:" + typeof Credentials);
                console.log("call:" + Credentials({ a: 1 }).id);
                console.log("named:" + named);
                console.log("ns-default-is-the-same:" + (ns.default === Credentials));
            "#,
        )
        .unwrap();
        let stdout = bundle_and_run(directory.path());
        assert_eq!(
            stdout.lines().collect::<Vec<_>>(),
            [
                "typeof:function",
                "call:credentials",
                "named:named-val",
                "ns-default-is-the-same:true",
            ],
            "{stdout}"
        );
    }

    /// The negation of the rule above, so it stays a rule and not a guess: a CommonJS
    /// module with NO `__esModule` marker keeps Node's semantics — a default import
    /// binds `module.exports`, even when the object happens to carry a `default` key.
    #[test]
    fn a_default_import_of_an_unmarked_commonjs_module_still_binds_module_exports() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("plain.cjs"),
            "module.exports = { default: \"inner\", other: \"o\" };\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import plain from \"./plain.cjs\";\nconsole.log(\"default:\" + JSON.stringify(plain));\n",
        )
        .unwrap();
        assert_eq!(
            bundle_and_run(directory.path()),
            "default:{\"default\":\"inner\",\"other\":\"o\"}\n",
        );
    }

    /// `serverExternalPackages` (next.config): a listed package is NOT bundled into a
    /// server graph — it stays a runtime `require` from `node_modules`. Apps use the
    /// list precisely because bundling the package fails, so a build that ignores it
    /// turns working configuration into a fatal error: cal.com externalizes
    /// `rest-facade`, whose `require('superagent-proxy')` sits behind a runtime `if`
    /// and names a package that is deliberately not installed.
    #[test]
    fn a_server_external_package_is_not_bundled_and_its_own_imports_are_not_resolved() {
        let directory = tempdir().unwrap();
        let package = directory.path().join("node_modules/rest-facade");
        fs::create_dir_all(&package).unwrap();
        fs::write(package.join("package.json"), r#"{"name":"rest-facade","main":"index.js"}"#).unwrap();
        // The shape that makes the list necessary: an import of a package that is not
        // installed, reached only on a branch the app never takes.
        fs::write(
            package.join("index.js"),
            "exports.Client = function (o) { if (o.proxy) require('superagent-proxy'); };\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { Client } from \"rest-facade\";\nexport const c = Client;\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");

        // Bundled (the default), the uninstalled transitive import is a fatal diagnostic.
        let (_, update) = Bundler::discover(&entry).unwrap();
        assert!(
            update.diagnostics.iter().any(|d| d.is_fatal()
                && matches!(d.kind, DiagnosticKind::UnresolvedImport { .. })
                && d.message.contains("superagent-proxy")),
            "without the list the uninstalled dependency is fatal: {:?}",
            update.diagnostics,
        );

        // Listed as a server external, the package is never resolved at all — so
        // nothing inside it can fail the build, and it is not a graph module.
        let config = BuildConfig {
            target: Target::Server,
            server_external_packages: vec!["rest-facade".to_string()],
            ..BuildConfig::default()
        };
        let (bundler, update) = Bundler::discover_direct_with_config(&entry, &config).unwrap();
        assert!(
            update.diagnostics.iter().all(|d| !d.is_fatal()),
            "an externalized package cannot fail the build: {:?}",
            update.diagnostics,
        );
        assert!(
            !bundler.ids.iter().any(|id| id.contains("rest-facade")),
            "the external must not be a graph module: {:?}",
            bundler.ids,
        );
    }

    /// A CLIENT graph must ignore the list: a browser has no `node_modules` to require
    /// from at runtime, so externalizing there would emit a chunk that dies on the
    /// throw-on-use stub with a zero build exit code.
    #[test]
    fn a_server_external_package_is_still_bundled_for_the_browser() {
        let directory = tempdir().unwrap();
        let package = directory.path().join("node_modules/jose");
        fs::create_dir_all(&package).unwrap();
        fs::write(package.join("package.json"), r#"{"name":"jose","main":"index.js"}"#).unwrap();
        fs::write(package.join("index.js"), "exports.sign = () => \"signed\";\n").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { sign } from \"jose\";\nexport const s = sign;\n",
        )
        .unwrap();
        let config = BuildConfig {
            target: Target::Client,
            server_external_packages: vec!["jose".to_string()],
            ..BuildConfig::default()
        };
        let (bundler, update) =
            Bundler::discover_direct_with_config(&directory.path().join("entry.js"), &config).unwrap();
        assert!(update.diagnostics.iter().all(|d| !d.is_fatal()), "{:?}", update.diagnostics);
        assert!(
            bundler.ids.iter().any(|id| id.contains("jose")),
            "the browser graph still bundles it: {:?}",
            bundler.ids,
        );
    }

    /// The interop namespace copies `module.exports`' keys at wrap time, which
    /// in an ESM<->CJS cycle is a PARTIALLY populated object. A key the module
    /// assigns after that point must still be readable through a named import
    /// rather than being frozen out (or, worse, reported as not provided).
    #[test]
    fn a_commonjs_export_assigned_after_the_interop_wrap_is_still_visible() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("legacy.cjs"),
            "exports.early = \"early\";\nrequire(\"./esm.js\");\nexports.late = \"late\";\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("esm.js"),
            "import { early, late } from \"./legacy.cjs\";\nexport function read() { return early + \"/\" + late; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import \"./legacy.cjs\";\nimport { read } from \"./esm.js\";\nconsole.log(\"read:\" + read());\n",
        )
        .unwrap();

        assert_eq!(bundle_and_run(directory.path()), "read:early/late\n");
    }

    /// One CommonJS module has exactly one interop namespace: re-running the
    /// interop over the same `module.exports` (`export * as ns from` re-runs it
    /// on every read) must return the same object, and running it over a
    /// namespace it already produced must be a no-op instead of nesting a
    /// second `default` around it.
    #[test]
    fn the_commonjs_interop_namespace_is_stable_and_idempotent() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("legacy.cjs"),
            "exports.value = \"legacy-value\";\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("hub.js"),
            "export * as legacy from \"./legacy.cjs\";\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            r#"
                import * as hub from "./hub.js";
                import * as direct from "./legacy.cjs";
                console.log("stable:" + (hub.legacy === hub.legacy));
                console.log("shared:" + (hub.legacy === direct));
                console.log("value:" + hub.legacy.value);
                console.log("not-nested:" + (hub.legacy.default.default === undefined));
            "#,
        )
        .unwrap();

        assert_eq!(
            bundle_and_run(directory.path()),
            "stable:true\nshared:true\nvalue:legacy-value\nnot-nested:true\n"
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

        if node_command().arg("--version").output().is_ok() {
            let executed = node_command().arg(&output).output().unwrap();
            assert!(
                executed.status.success(),
                "{}",
                String::from_utf8_lossy(&executed.stderr)
            );
            assert_eq!(String::from_utf8_lossy(&executed.stdout), format!("{url}\n"));
        }
    }

    /// A module that is BOTH named-imported and bare-`require`d by the same
    /// importer keeps its whole-module demand: `require("m")` hands out
    /// `module.exports` wholesale, so the import statement's named list must not
    /// downgrade the demand and shake off exports the require observably reads.
    /// This is exactly the shape of the next adapter's lazy island pins (a
    /// require thunk beside a named import of `control-boundary`), where the
    /// downgrade shook off the island's `default` export and broke hydration.
    #[test]
    fn a_bare_require_beside_a_named_import_keeps_every_export() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("m.js"),
            "export default function island() { return \"DEFAULT\"; }\n\
             export const named = \"NAMED\";\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { named } from './m.js';\n\
             const pins = [() => require('./m.js')];\n\
             globalThis.__pins = pins;\n\
             console.log(named, typeof pins[0]().default === 'function' ? pins[0]().default() : 'MISSING');\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();
        if node_command().arg("--version").output().is_ok() {
            let executed = node_command().arg(&output).output().unwrap();
            assert!(
                executed.status.success(),
                "{}",
                String::from_utf8_lossy(&executed.stderr)
            );
            assert_eq!(
                String::from_utf8_lossy(&executed.stdout),
                "NAMED DEFAULT\n"
            );
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
        if node_command().arg("--version").output().is_ok() {
            let executed = node_command().arg(&output).output().unwrap();
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
    fn worker_query_import_emits_a_worker_chunk_and_references_its_public_url() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("job.js"),
            "self.onmessage = (event) => self.postMessage(event.data * 2);\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import JobWorker from './job.js?worker';\nconst worker = new JobWorker();\nconsole.log(worker);\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();

        let bundle = fs::read_to_string(&output).unwrap();
        // The worker URL placeholder must be fully substituted — a leftover
        // `__diffpack_worker__…__` would 404 at runtime.
        assert!(
            !bundle.contains("__diffpack_worker__"),
            "worker placeholder left in bundle:\n{bundle}"
        );
        // The bundle spawns a `Worker` at the emitted chunk's public URL.
        let url = bundle
            .lines()
            .find_map(|line| line.find("/assets/job-").map(|start| &line[start..]))
            .and_then(|rest| rest.split('"').next())
            .expect("bundle should reference the worker chunk url");
        assert!(url.ends_with(".worker.js"), "{url}");

        // The self-contained worker chunk is emitted next to the bundle and
        // carries the entry's code.
        let worker_path = directory
            .path()
            .join("dist/assets")
            .join(url.trim_start_matches("/assets/"));
        assert!(worker_path.is_file(), "missing {}", worker_path.display());
        let worker_code = fs::read_to_string(&worker_path).unwrap();
        assert!(
            worker_code.contains("postMessage"),
            "worker chunk should bundle the entry code:\n{worker_code}"
        );
    }

    #[test]
    fn worker_inline_combo_reports_a_specific_unimplemented_error() {
        let error = match synthesize_worker(&ResourceId::parse("/abs/job.js?worker&inline")) {
            Err(error) => error,
            Ok(_) => panic!("?worker&inline should be refused"),
        };
        assert!(error.contains("?worker&inline"), "{error}");
        assert!(!error.contains("No such file or directory"), "{error}");
    }

    #[test]
    fn inline_query_import_embeds_the_asset_as_a_data_uri() {
        let directory = tempdir().unwrap();
        let png: &[u8] = b"\x89PNG\r\n\x1a\nfake-png-bytes";
        fs::write(directory.path().join("pixel.png"), png).unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import pixel from './pixel.png?inline';\nconsole.log(pixel);\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();

        let bundle = fs::read_to_string(&output).unwrap();
        let expected = format!("data:image/png;base64,{}", base64_encode(png));
        assert!(bundle.contains(&expected), "bundle should embed the data URI:\n{bundle}");
        // An inlined asset emits no separate file.
        let assets_dir = directory.path().join("dist/assets");
        assert!(
            !assets_dir.exists() || fs::read_dir(&assets_dir).unwrap().next().is_none(),
            "?inline must not emit a separate asset file"
        );
    }

    #[test]
    fn wasm_init_import_emits_the_module_and_a_default_initializer() {
        let directory = tempdir().unwrap();
        // A minimal well-formed WebAssembly module: the `\0asm` magic + version 1.
        let wasm: &[u8] = &[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        fs::write(directory.path().join("add.wasm"), wasm).unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import init from './add.wasm?init';\ninit().then((instance) => console.log(instance));\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();

        let bundle = fs::read_to_string(&output).unwrap();
        // The instantiation helper is inlined.
        assert!(bundle.contains("WebAssembly"), "helper missing:\n{bundle}");
        assert!(bundle.contains("instantiate"), "helper missing:\n{bundle}");

        // The `.wasm` payload takes the content-hashed asset pipeline (default
        // inline limit is 0, so it is a real file, not a data URI) and the
        // initializer closes over its URL.
        let url = bundle
            .lines()
            .find_map(|line| line.find("/assets/add-").map(|start| &line[start..]))
            .and_then(|rest| rest.split('"').next())
            .expect("bundle should reference the hashed wasm url");
        assert!(url.ends_with(".wasm"), "{url}");
        let wasm_path = directory
            .path()
            .join("dist/assets")
            .join(url.trim_start_matches("/assets/"));
        assert_eq!(fs::read(&wasm_path).unwrap(), wasm);
    }

    #[test]
    fn wasm_init_inlines_a_small_module_as_a_data_uri() {
        let directory = tempdir().unwrap();
        let wasm: &[u8] = &[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        fs::write(directory.path().join("tiny.wasm"), wasm).unwrap();
        // A generous inline limit forces the payload into a `data:` URI.
        // Guard: `?init` only applies to `.wasm`.
        assert!(synthesize_wasm_init(&ResourceId::parse("tiny.js?init"), "/", 4096).is_err());
        let path = directory.path().join("tiny.wasm");
        let module = synthesize_wasm_init(
            &ResourceId::parse(&format!("{}?init", path.display())),
            "/",
            4096,
        )
        .unwrap();
        assert!(module.assets.is_empty(), "small wasm should inline, not emit a file");
        assert!(
            module.code.contains("data:application/wasm;base64,"),
            "small wasm should be a data URI:\n{}",
            module.code
        );
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
    fn an_unrecognized_loader_query_reports_a_specific_error() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("thing.js"), "export const x = 1;").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import c from './thing.js?mystery';\nconsole.log(c);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let error = match Bundler::discover_direct(&entry) {
            Ok(_) => panic!("an unimplemented loader must fail the build, not silently succeed"),
            Err(error) => error,
        };
        assert!(
            error.contains("unrecognized loader query `?mystery`"),
            "{error}"
        );
        assert!(!error.contains("No such file or directory"), "{error}");
    }

    #[test]
    fn a_tsr_split_query_on_a_non_route_file_reports_a_specific_error() {
        // `?tsr-split` is implemented, but only for route files. Asking a plain
        // module to produce a split module is a clear error, not a silent empty
        // module or a filesystem crash.
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("thing.js"), "export const x = 1;").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import c from './thing.js?tsr-split=component';\nconsole.log(c);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let error = match Bundler::discover_direct(&entry) {
            Ok(_) => panic!("a tsr-split on a non-route file must fail the build"),
            Err(error) => error,
        };
        assert!(
            error.contains("not a splittable route file"),
            "{error}"
        );
        assert!(!error.contains("No such file or directory"), "{error}");
    }

    /// Every wording a reader would chase down the WRONG path: a JSX syntax error
    /// in their own file, or a resolution failure. An unhandled source is neither.
    fn assert_not_misreported(error: &str) {
        for misleading in [
            "Unexpected JSX expression",
            "cannot resolve",
            "unresolved",
            "npm install",
            "No such file or directory",
        ] {
            assert!(!error.contains(misleading), "{misleading}: {error}");
        }
    }

    #[test]
    fn a_vue_component_whose_compiler_is_missing_names_the_package_not_a_jsx_error() {
        // A `.vue` file is not JavaScript. Parsing it as JavaScript reports
        // `Unexpected JSX expression` on the app's `<template>`, blaming the app
        // for diffpack's own gap. It is compiled by the APP's OWN
        // `@vue/compiler-sfc`; this fixture project has no `node_modules` at all,
        // so the compile must fail loudly, naming the file and the package —
        // never fall back to reading the component as JavaScript.
        // (Requires `node` on PATH, as every diffpack build already does.)
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("App.vue"),
            "<script setup lang=\"ts\">\nconst greeting = 'hi';\n</script>\n\n\
             <template>\n  <h1>{{ greeting }}</h1>\n</template>\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import App from './App.vue';\nconsole.log(App);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let error = match Bundler::discover_direct(&entry) {
            Ok(_) => panic!("a `.vue` component has no compiler here; it must fail the build"),
            Err(error) => error,
        };
        assert!(error.contains("App.vue"), "{error}");
        assert!(error.contains("Vue single-file component"), "{error}");
        assert!(error.contains("@vue/compiler-sfc"), "{error}");
        assert_not_misreported(&error);
    }

    #[test]
    fn a_svelte_component_whose_compiler_is_missing_names_the_package() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("App.svelte"),
            "<script lang=\"ts\">\n  let count = 0;\n</script>\n\n<h1>{count}</h1>\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import App from './App.svelte';\nconsole.log(App);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let error = match Bundler::discover_direct(&entry) {
            Ok(_) => panic!("a `.svelte` component has no compiler here; it must fail the build"),
            Err(error) => error,
        };
        assert!(error.contains("App.svelte"), "{error}");
        assert!(error.contains("Svelte component"), "{error}");
        assert!(error.contains("svelte/compiler"), "{error}");
        assert_not_misreported(&error);
    }

    /// A build configured like a Vite project: a real project root (so
    /// root-absolute and `public/` imports resolve) and the client target.
    fn vite_like_config(root: &Path, aliases: Vec<(String, String)>) -> BuildConfig {
        BuildConfig {
            base: "/".to_string(),
            browser_process_shim: false,
            asset_inline_limit: 0,
            aliases,
            conditions: vec!["module".into(), "browser".into()],
            main_fields: Vec::new(),
            virtual_modules: Vec::new(),
            target: Target::Client,
            server_external_packages: Vec::new(),
            import_meta_env: None,
            import_meta_glob: None,
            defines: Vec::new(),
            hmr: false,
            scss: crate::sass::ScssOptions::default(),
            image_import_shape: ImageImportShape::Url,
            css_preprocess: CssPreprocess {
                root: Some(root.to_path_buf()),
                postcss: None,
            },
            jsx_extensions: crate::parser::JsxExtensions::JsxAndTsxOnly,
            jsx: crate::transform::JsxConfig::default(),
            source_maps: false,
        }
    }

    #[test]
    fn a_root_absolute_import_of_a_public_file_is_its_url_not_an_emitted_asset() {
        // Vite: `import icons from "/icons.svg"` is `<root>/icons.svg`, not the
        // filesystem path `/icons.svg`. With no such file in the root but one in
        // `public/`, the import is the file's PUBLIC URL — `public/` is copied to
        // the site root verbatim, so hashing and re-emitting it would mint a
        // second copy at a URL the app's own build never produces.
        // (Vue's SFC compiler emits exactly this import for a `<use href="/icons.svg#x">`.)
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::create_dir_all(root.join("public")).unwrap();
        fs::write(root.join("public/icons.svg"), "<svg/>").unwrap();
        fs::write(
            root.join("entry.js"),
            "import icons from '/icons.svg';\nconsole.log(icons);\n",
        )
        .unwrap();
        let entry = root.join("entry.js");
        let config = vite_like_config(root, Vec::new());
        let (bundler, update) = Bundler::discover_direct_with_config(&entry, &config).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = root.join("dist/bundle.js");
        bundler.emit(&reachable, &output).unwrap();
        let code = fs::read_to_string(&output).unwrap();
        assert!(code.contains("\"/icons.svg\"") || code.contains("'/icons.svg'"), "{code}");
        // No hashed copy: the public file is served from the site root as-is.
        assert!(!root.join("dist/assets").exists(), "a public file must not be re-emitted");
    }

    /// Write a package under `<root>/node_modules/<name>/` from `(relative path,
    /// contents)` pairs. `package.json` is one of the pairs, so the test owns the
    /// whole manifest (`browser`, `exports`, …).
    fn write_package_files(root: &Path, name: &str, files: &[(&str, &str)]) {
        let base = root.join("node_modules").join(name);
        for (relative, contents) in files {
            let path = base.join(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, contents).unwrap();
        }
    }

    #[test]
    fn an_object_browser_field_remaps_a_packages_own_relative_import() {
        // The classic pre-`exports` substitution map in its OBJECT form: keys are
        // paths RELATIVE TO THE PACKAGE ROOT, and they rewrite the package's own
        // internal `./node.js` import — not just its entry point. axios ships
        // exactly this shape to keep `lib/adapters/http.js` (which imports `http`,
        // `https`, `zlib`, …) out of browser bundles. Honouring only the string
        // form drags the Node implementation, and every Node built-in it touches,
        // into the client graph.
        let directory = tempdir().unwrap();
        let root = directory.path();
        write_package_files(
            root,
            "swappable",
            &[
                (
                    "package.json",
                    r#"{
                      "name": "swappable",
                      "version": "1.0.0",
                      "type": "module",
                      "main": "./index.js",
                      "exports": { ".": "./index.js" },
                      "browser": { "./lib/node.js": "./lib/browser.js" }
                    }"#,
                ),
                ("index.js", "export { impl } from './lib/node.js';\n"),
                (
                    "lib/node.js",
                    "import zlib from 'zlib';\nexport const impl = 'NODE_IMPL' + typeof zlib;\n",
                ),
                ("lib/browser.js", "export const impl = 'BROWSER_IMPL';\n"),
            ],
        );
        fs::write(
            root.join("entry.js"),
            "import { impl } from 'swappable';\nconsole.log(impl);\n",
        )
        .unwrap();
        let entry = root.join("entry.js");
        let config = vite_like_config(root, Vec::new());
        let (bundler, update) = Bundler::discover_direct_with_config(&entry, &config).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = root.join("dist/bundle.js");
        bundler.emit(&reachable, &output).unwrap();
        let code = fs::read_to_string(&output).unwrap();
        assert!(!code.contains("NODE_IMPL"), "the node variant must not be bundled: {code}");
        assert_eq!(run_node(&output), "BROWSER_IMPL\n");
    }

    #[test]
    fn a_false_browser_field_entry_stubs_a_module_out_of_the_browser_graph() {
        // `"browser": { "./lib/node.js": false }` means "this module is empty in a
        // browser". webpack/Vite substitute an empty module; leaving the real one
        // in place pulls its Node built-ins into the client bundle.
        let directory = tempdir().unwrap();
        let root = directory.path();
        write_package_files(
            root,
            "stubbable",
            &[
                (
                    "package.json",
                    r#"{
                      "name": "stubbable",
                      "version": "1.0.0",
                      "type": "module",
                      "main": "./index.js",
                      "exports": { ".": "./index.js" },
                      "browser": { "./lib/node.js": false }
                    }"#,
                ),
                (
                    "index.js",
                    "import * as node from './lib/node.js';\nexport const impl = typeof node.impl;\n",
                ),
                (
                    "lib/node.js",
                    "import zlib from 'zlib';\nexport const impl = 'NODE_IMPL' + typeof zlib;\n",
                ),
            ],
        );
        fs::write(
            root.join("entry.js"),
            "import { impl } from 'stubbable';\nconsole.log(impl);\n",
        )
        .unwrap();
        let entry = root.join("entry.js");
        let config = vite_like_config(root, Vec::new());
        let (bundler, update) = Bundler::discover_direct_with_config(&entry, &config).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = root.join("dist/bundle.js");
        bundler.emit(&reachable, &output).unwrap();
        let code = fs::read_to_string(&output).unwrap();
        assert!(!code.contains("NODE_IMPL"), "the stubbed module must not be bundled: {code}");
        // webpack's semantics: the excluded module is an object with nothing on it.
        assert_eq!(run_node(&output), "undefined\n");
    }

    #[test]
    fn a_try_guarded_require_of_an_uninstalled_package_is_a_warning_not_a_build_error() {
        // `try { require("accelerator") } catch {}` is how packages with native or
        // platform-specific accelerators declare an optional dependency (`ws` ->
        // bufferutil/utf-8-validate, `pg` -> pg-native, `sharp` -> @img/*, jsdom ->
        // canvas). Node throws MODULE_NOT_FOUND at that `require` and the `catch`
        // supplies the fallback, so the program is CORRECT with the package absent.
        // Failing the build rejects code that runs.
        let directory = tempdir().unwrap();
        let root = directory.path();
        write_package_files(
            root,
            "guarded",
            &[
                (
                    "package.json",
                    r#"{ "name": "guarded", "version": "1.0.0", "main": "./index.js" }"#,
                ),
                (
                    "index.js",
                    "let fast;\ntry { fast = require('accelerator'); } catch { fast = null; }\n\
                     module.exports.impl = fast ? 'FAST' : 'FALLBACK_OK';\n",
                ),
            ],
        );
        fs::write(
            root.join("entry.js"),
            "import { impl } from 'guarded';\nconsole.log(impl);\n",
        )
        .unwrap();
        let entry = root.join("entry.js");
        let config = vite_like_config(root, Vec::new());
        let (bundler, update) = Bundler::discover_direct_with_config(&entry, &config).unwrap();
        assert!(
            update.diagnostics.iter().all(|diagnostic| !diagnostic.is_fatal()),
            "a guarded optional require must not be fatal: {:?}",
            update.diagnostics
        );
        // Reported, though: an omission nobody is told about is a silent fallback.
        assert!(
            update.diagnostics.iter().any(|diagnostic| matches!(
                &diagnostic.kind,
                DiagnosticKind::OptionalDependencyMissing { specifier, .. }
                    if specifier == "accelerator"
            )),
            "the omission must still be reported: {:?}",
            update.diagnostics
        );
        let reachable = bundler.reachable_modules_direct();
        let output = root.join("dist/bundle.js");
        bundler.emit(&reachable, &output).unwrap();
        // Node semantics preserved end to end: the require throws, the catch runs.
        assert_eq!(run_node(&output), "FALLBACK_OK\n");
    }

    #[test]
    fn an_unguarded_require_of_the_same_package_stays_a_fatal_build_error() {
        // The counterpart that keeps the rule honest. One reference outside a `try`
        // means some path really does need the module, so its absence still breaks
        // the artifact — a typo inside a package must not be laundered into a
        // warning just because the same name also appears in a guarded require.
        let directory = tempdir().unwrap();
        let root = directory.path();
        write_package_files(
            root,
            "unguarded",
            &[
                (
                    "package.json",
                    r#"{ "name": "unguarded", "version": "1.0.0", "main": "./index.js" }"#,
                ),
                (
                    "index.js",
                    "let fast;\ntry { fast = require('accelerator'); } catch { fast = null; }\n\
                     const always = require('accelerator');\nmodule.exports.impl = always;\n",
                ),
            ],
        );
        fs::write(
            root.join("entry.js"),
            "import { impl } from 'unguarded';\nconsole.log(impl);\n",
        )
        .unwrap();
        let entry = root.join("entry.js");
        let config = vite_like_config(root, Vec::new());
        let (bundler, update) = Bundler::discover_direct_with_config(&entry, &config).unwrap();
        let _ = &bundler;
        assert!(
            update.diagnostics.iter().any(|diagnostic| matches!(
                &diagnostic.kind,
                DiagnosticKind::UnresolvedImport { specifier, .. } if specifier == "accelerator"
            )),
            "an unguarded reference must stay fatal: {:?}",
            update.diagnostics
        );
    }

    #[test]
    fn a_foreign_runtimes_scheme_specifier_is_classified_like_a_node_builtin() {
        // `node:fs` already means "the host provides this". Every other runtime uses
        // the same reserved shape for its own built-ins, and diffpack's rule was
        // accidentally Node-only — so `cloudflare:sockets` (imported by pg-cloudflare,
        // which `pg` pulls in) was reported as a missing package with the impossible
        // advice `npm install cloudflare:sockets`.
        assert_eq!(host_provided_scheme("cloudflare:sockets"), Some("cloudflare"));
        assert_eq!(host_provided_scheme("bun:ffi"), Some("bun"));
        assert_eq!(host_provided_scheme("node:fs"), Some("node"));
        // Resource URLs address bytes rather than naming a host module; diffpack
        // cannot load any of them, so they must keep failing.
        assert_eq!(host_provided_scheme("https://esm.sh/react"), None);
        assert_eq!(host_provided_scheme("data:text/javascript,export{}"), None);
        assert_eq!(host_provided_scheme("file:///tmp/x.js"), None);
        // Ordinary specifiers, and a Windows drive path, are not schemes.
        assert_eq!(host_provided_scheme("react"), None);
        assert_eq!(host_provided_scheme("./local.js"), None);
        assert_eq!(host_provided_scheme("@scope/pkg"), None);
        assert_eq!(host_provided_scheme("C:/project/src/x.js"), None);
        assert_eq!(host_provided_scheme("cloudflare:"), None);
    }

    #[test]
    fn a_foreign_runtime_module_is_external_on_a_server_graph_and_fatal_on_a_client_one() {
        // The consequence of the classification above, at both targets.
        let directory = tempdir().unwrap();
        let root = directory.path();
        // pg-cloudflare's shape: a package whose socket implementation reaches for the
        // Workers runtime module, pulled in unconditionally by its parent (`pg`).
        write_package_files(
            root,
            "workers-socket",
            &[
                (
                    "package.json",
                    r#"{ "name": "workers-socket", "version": "1.0.0", "main": "./index.js" }"#,
                ),
                (
                    "index.js",
                    "module.exports.connect = async () => (await import('cloudflare:sockets')).connect;\n",
                ),
            ],
        );
        fs::write(
            root.join("entry.js"),
            "import { connect } from 'workers-socket';\nexport { connect };\n",
        )
        .unwrap();
        let entry = root.join("entry.js");

        let mut server = vite_like_config(root, Vec::new());
        server.target = Target::Server;
        server.conditions = vec!["node".into()];
        let (_, update) = Bundler::discover_direct_with_config(&entry, &server).unwrap();
        assert!(
            update.diagnostics.iter().all(|diagnostic| !diagnostic.is_fatal()),
            "a host-provided module must not fail a server build: {:?}",
            update.diagnostics
        );
        assert!(
            update.diagnostics.iter().any(|diagnostic| matches!(
                &diagnostic.kind,
                DiagnosticKind::HostProvidedModule { specifier, .. }
                    if specifier == "cloudflare:sockets"
            )),
            "the external must still be reported: {:?}",
            update.diagnostics
        );

        // A browser has no host to provide it, so it stays fatal — same as `node:fs`.
        let client = vite_like_config(root, Vec::new());
        let (_, update) = Bundler::discover_direct_with_config(&entry, &client).unwrap();
        assert!(
            update.diagnostics.iter().any(|diagnostic| diagnostic.is_fatal()),
            "a browser graph has no host runtime: {:?}",
            update.diagnostics
        );
    }

    #[test]
    fn a_root_absolute_import_prefers_a_file_in_the_project_root() {
        // `/lib/util.js` with `<root>/lib/util.js` present is that module, not a
        // public URL — Vite resolves root-relative first.
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::create_dir_all(root.join("lib")).unwrap();
        fs::write(root.join("lib/util.js"), "export const value = 'ROOT_RELATIVE_OK';\n").unwrap();
        fs::write(
            root.join("entry.js"),
            "import { value } from '/lib/util.js';\nconsole.log(value);\n",
        )
        .unwrap();
        let entry = root.join("entry.js");
        let config = vite_like_config(root, Vec::new());
        let (bundler, update) = Bundler::discover_direct_with_config(&entry, &config).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = root.join("dist/bundle.js");
        bundler.emit(&reachable, &output).unwrap();
        let code = fs::read_to_string(&output).unwrap();
        assert!(code.contains("ROOT_RELATIVE_OK"), "{code}");
    }

    #[test]
    fn a_dedupe_alias_still_resolves_a_subpath_through_the_package_exports_map() {
        // Vite's `resolve.dedupe` pins a package to `<root>/node_modules/<pkg>`,
        // which diffpack carries as a directory alias. A SUBPATH cannot be
        // answered by joining onto that directory: `svelte/internal/client` is a
        // key in the package's `exports` map, not a file at that path, so the
        // join produced a path that does not exist and the build failed on a
        // package that is installed.
        let directory = tempdir().unwrap();
        let root = directory.path();
        let package = root.join("node_modules/widget");
        fs::create_dir_all(package.join("src/internal")).unwrap();
        fs::write(
            package.join("package.json"),
            "{\"name\":\"widget\",\"exports\":{\".\":\"./src/index.js\",\
             \"./internal/client\":\"./src/internal/client-impl.js\"}}",
        )
        .unwrap();
        fs::write(package.join("src/index.js"), "export const main = 1;\n").unwrap();
        fs::write(
            package.join("src/internal/client-impl.js"),
            "export const internalValue = 'EXPORTS_SUBPATH_OK';\n",
        )
        .unwrap();
        fs::write(
            root.join("entry.js"),
            "import { internalValue } from 'widget/internal/client';\nconsole.log(internalValue);\n",
        )
        .unwrap();
        let entry = root.join("entry.js");
        let aliases = vec![(
            "widget".to_string(),
            package.to_string_lossy().into_owned(),
        )];
        let config = vite_like_config(root, aliases);
        let (bundler, update) = Bundler::discover_direct_with_config(&entry, &config).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = root.join("dist/bundle.js");
        bundler.emit(&reachable, &output).unwrap();
        let code = fs::read_to_string(&output).unwrap();
        assert!(code.contains("EXPORTS_SUBPATH_OK"), "{code}");
    }

    #[test]
    fn an_extension_no_loader_handles_is_named_not_parsed_as_javascript() {
        // diffpack does not know what a `.graphql` file is, and must say exactly
        // that rather than invent a compiler for it or parse it as JavaScript.
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("schema.graphql"),
            "type Query { hello: String }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import schema from './schema.graphql';\nconsole.log(schema);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let error = match Bundler::discover_direct(&entry) {
            Ok(_) => panic!("no loader handles `.graphql`; it must fail the build"),
            Err(error) => error,
        };
        assert!(error.contains("no loader handles the `.graphql` extension"), "{error}");
        assert!(error.contains("./schema.graphql?raw"), "{error}");
        assert!(!error.contains("compiler"), "{error}");
        assert_not_misreported(&error);
    }

    #[test]
    fn a_native_addon_is_reported_where_it_is_loaded_not_where_it_is_resolved() {
        // A `.node` addon resolves perfectly well. Failing it inside the resolver
        // printed `cannot resolve ...` plus `install it: npm install ...` for a
        // file sitting right there on disk.
        let directory = tempdir().unwrap();
        let package = directory.path().join("node_modules/native-addon");
        fs::create_dir_all(&package).unwrap();
        fs::write(
            package.join("package.json"),
            "{\"name\":\"native-addon\",\"main\":\"index.node\"}",
        )
        .unwrap();
        fs::write(package.join("index.node"), [0x7f, b'E', b'L', b'F']).unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import addon from 'native-addon';\nconsole.log(addon);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let error = match Bundler::discover_direct(&entry) {
            Ok(_) => panic!("native code cannot go in a JavaScript bundle; it must fail the build"),
            Err(error) => error,
        };
        assert!(error.contains("index.node"), "{error}");
        assert!(error.contains("prebuilt native addon"), "{error}");
        assert_not_misreported(&error);
    }

    #[test]
    fn a_vue_file_still_loads_through_the_raw_loader() {
        // The query check runs BEFORE the extension table, so `?raw`/`?url` remain
        // the escape hatch for any extension diffpack cannot compile itself.
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("App.vue"),
            "<template><h1>hi</h1></template>\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import source from './App.vue?raw';\nconsole.log(source);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();
        let code = fs::read_to_string(&output).unwrap();
        assert!(code.contains("<template><h1>hi</h1></template>"), "{code}");
    }

    #[test]
    fn mts_cts_and_extensionless_modules_still_build_as_javascript() {
        // The extension table is an ALLOW-list for JavaScript, so it must not
        // reject the JS-family extensions the resolver never adds implicitly
        // (`.mts`/`.cts`) or the extensionless files `node_modules` is full of.
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("a.mts"), "export const a = 1;\n").unwrap();
        fs::write(directory.path().join("b.cts"), "export const b = 2;\n").unwrap();
        fs::create_dir_all(directory.path().join("bin")).unwrap();
        fs::write(directory.path().join("bin/cli"), "export const c = 3;\n").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { a } from './a.mts';\nimport { b } from './b.cts';\n\
             import { c } from './bin/cli';\nconsole.log(a + b + c);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let output = directory.path().join("dist/bundle.js");
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();
    }

    #[test]
    fn a_source_error_is_not_reported_as_a_dangling_reference() {
        // "Dangling references" describes what an UNRESOLVED IMPORT leaves behind.
        // Saying it for a module that never compiled points the reader at an
        // import that is perfectly fine.
        let source_only = [Diagnostic {
            kind: DiagnosticKind::Source { fatal: true },
            message: "App.vue: `.vue` is a Vue single-file component".into(),
        }];
        let error = partition_diagnostics(&source_only, "page `index`").unwrap_err();
        assert!(!error.contains("dangling"), "{error}");
        assert!(error.contains("would not match the source"), "{error}");

        let unresolved_only = [Diagnostic {
            kind: DiagnosticKind::UnresolvedImport {
                specifier: "left-pad".into(),
                importer: PathBuf::from("entry.js"),
            },
            message: "cannot resolve \"left-pad\"".into(),
        }];
        let error = partition_diagnostics(&unresolved_only, "page `index`").unwrap_err();
        assert!(error.contains("dangling references"), "{error}");
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

        // The external require survives for the runtime to resolve. A static
        // import goes through `require.esm`, which calls that same `require` and
        // falls back to `__toESM` for a specifier the graph does not own.
        let bundle = fs::read_to_string(&output).unwrap();
        assert!(bundle.contains("require.esm(\"node:path\")"), "{bundle}");

        if node_command().arg("--version").output().is_ok() {
            let executed = node_command().arg(&output).output().unwrap();
            assert!(
                executed.status.success(),
                "{}",
                String::from_utf8_lossy(&executed.stderr)
            );
            assert_eq!(String::from_utf8_lossy(&executed.stdout), "c.txt:nl\n");
        }
    }

    /// The browser build's node-builtin stub exists so that dead server code
    /// which leaked into the client graph still LOADS, and throws a
    /// specifically-named error only when it actually calls into the built-in.
    /// A named import is a read like any other: it must hand back the stub, not
    /// trip `__import`'s "does not provide an export" check — the stub is a
    /// Proxy whose shape is unknowable, so absence there proves nothing.
    #[test]
    fn a_named_import_of_a_node_builtin_in_a_browser_build_stubs_instead_of_throwing() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("entry.js"),
            r#"
                import { readFileSync } from "node:fs";
                console.log("loaded:" + (typeof readFileSync));
                try {
                  readFileSync("/etc/hosts");
                } catch (error) {
                  console.log("called:" + error.message);
                }
            "#,
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.mjs");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler
            .emit_with_options(
                &reachable,
                &output,
                EmitOptions {
                    format: ModuleFormat::BrowserEsm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();

        let executed = node_command().arg(&output).output().unwrap();
        assert!(
            executed.status.success(),
            "{}",
            String::from_utf8_lossy(&executed.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&executed.stdout),
            "loaded:function\ncalled:node builtin node:fs is not available in the browser\n"
        );
    }

    /// A CommonJS module has exactly ONE ES namespace, whatever `module.exports`
    /// happens to be.
    ///
    /// `export * as ns from "cjs"` compiles to a getter, so the interop re-runs on
    /// every read of `ns`. `__cjsNamespaces` keys the wrapper by the `module.exports`
    /// object, which covers nothing when `module.exports = 42`: a WeakMap takes no
    /// primitive key, so every read minted a fresh namespace and `ns.legacy ===
    /// ns.legacy` was `false` where Node (and rolldown) say `true`.
    ///
    /// The identity that exists for every value shape is the MODULE, which is why a
    /// static import goes through `require.esm` (keyed by module id) rather than
    /// `__toESM(require(...))`. Caching by primitive VALUE instead would be a second
    /// wrong answer, and the second half of this test is what forbids it: two
    /// modules that each `module.exports = 42` are two namespaces.
    #[test]
    fn one_commonjs_module_has_one_namespace_even_when_its_exports_are_a_primitive() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::write(root.join("num.cjs"), "module.exports = 42;\n").unwrap();
        fs::write(root.join("other.cjs"), "module.exports = 42;\n").unwrap();
        fs::write(root.join("a.js"), "export * as legacy from \"./num.cjs\";\n").unwrap();
        fs::write(root.join("b.js"), "export * as legacy from \"./num.cjs\";\n").unwrap();
        fs::write(root.join("c.js"), "export * as legacy from \"./other.cjs\";\n").unwrap();
        fs::write(
            root.join("entry.js"),
            "import * as a from \"./a.js\";\n\
             import * as b from \"./b.js\";\n\
             import * as c from \"./c.js\";\n\
             console.log(\"stable:\" + (a.legacy === a.legacy));\n\
             console.log(\"shared:\" + (a.legacy === b.legacy));\n\
             console.log(\"distinct:\" + (a.legacy === c.legacy));\n\
             console.log(\"default:\" + a.legacy.default);\n",
        )
        .unwrap();

        let entry = root.join("entry.js");
        let output = root.join("dist/bundle.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();

        let executed = node_command().arg(&output).output().unwrap();
        assert!(
            executed.status.success(),
            "{}",
            String::from_utf8_lossy(&executed.stderr)
        );
        // Node's own answer for this program, unbundled.
        assert_eq!(
            String::from_utf8_lossy(&executed.stdout),
            "stable:true\nshared:true\ndistinct:false\ndefault:42\n"
        );
    }

    /// A Node built-in reached from a BROWSER graph is a FATAL build diagnostic,
    /// not a silent external. Leaving it external emits a `require` no browser can
    /// satisfy: the build exits 0 and the page dies. The same specifier on a
    /// SERVER graph stays external and is not a diagnostic at all — Node resolves
    /// it. The classifier alone cannot tell these apart, which is why
    /// `resolve_dependencies` takes the `Target`.
    #[test]
    fn a_node_builtin_is_fatal_in_a_browser_build_and_external_in_a_server_build() {
        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.js");
        fs::write(
            &entry,
            "import { format } from \"url\";\nconsole.log(format({}));\n",
        )
        .unwrap();
        let config = |target| BuildConfig {
            target,
            ..BuildConfig::default()
        };

        let (_bundler, client) =
            Bundler::discover_direct_with_config(&entry, &config(Target::Client)).unwrap();
        let fatal: Vec<_> = client
            .diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.is_fatal())
            .collect();
        assert_eq!(fatal.len(), 1, "{:?}", client.diagnostics);
        assert!(
            matches!(
                &fatal[0].kind,
                DiagnosticKind::NodeBuiltinInBrowser { specifier, .. } if specifier == "url"
            ),
            "{:?}",
            fatal[0].kind
        );
        // The message names the built-in AND the file that imported it: the fix is
        // to stop pulling that file into the client graph.
        assert!(fatal[0].message.contains("\"url\""), "{}", fatal[0].message);
        assert!(
            fatal[0].message.contains("entry.js"),
            "{}",
            fatal[0].message
        );
        // And it must stop the build, not warn.
        assert!(partition_diagnostics(&client.diagnostics, "client build").is_err());

        let (_bundler, server) =
            Bundler::discover_direct_with_config(&entry, &config(Target::Server)).unwrap();
        assert!(
            server
                .diagnostics
                .iter()
                .all(|diagnostic| !diagnostic.is_fatal()),
            "{:?}",
            server.diagnostics
        );
    }

    /// The browser `requireNative` fallback must not claim that an npm package is
    /// a "node builtin", and must not hand back a lazy stub for one. Every
    /// optional dependency in the ecosystem is loaded as
    /// `try { require(pkg) } catch {}`; returning a Proxy defeats the `catch`,
    /// smuggles the stub in as a real value, and throws later somewhere unrelated
    /// (this is exactly how `next-pages-framer-motion` died on
    /// `@emotion/is-prop-valid`). Node throws immediately for an absent module, so
    /// so do we — while a genuine Node built-in keeps the load-safe stub.
    #[test]
    fn a_non_builtin_runtime_require_throws_immediately_in_a_browser_build() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        // The specifier is assembled at runtime, exactly as framer-motion does it,
        // so the bundler never sees it as a static dependency and it reaches the
        // `requireNative` fallback.
        fs::write(
            directory.path().join("entry.js"),
            r#"
                const pkg = "@emotion/is-prop-" + "valid";
                let loaded = "fallback";
                try {
                  loaded = require(pkg).default;
                } catch (error) {
                  console.log("caught:" + error.message);
                }
                console.log("value:" + loaded);
            "#,
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.mjs");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler
            .emit_with_options(
                &reachable,
                &output,
                EmitOptions {
                    format: ModuleFormat::BrowserEsm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();

        let executed = node_command().arg(&output).output().unwrap();
        assert!(
            executed.status.success(),
            "{}",
            String::from_utf8_lossy(&executed.stderr)
        );
        let stdout = String::from_utf8_lossy(&executed.stdout);
        // It threw (so the app's own `catch` ran and its fallback survived) ...
        assert!(stdout.contains("caught:"), "{stdout}");
        assert!(stdout.contains("value:fallback"), "{stdout}");
        // ... and it did NOT call an npm package a node builtin.
        assert!(
            !stdout.contains("node builtin"),
            "an npm package must not be reported as a Node built-in: {stdout}"
        );
        assert!(
            stdout.contains("@emotion/is-prop-valid"),
            "the error must name the specifier: {stdout}"
        );
    }

    /// `__dirname`/`__filename` in a BROWSER bundle. Node's ESM entry defines them
    /// from `import.meta.url`, but a browser chunk has no location to derive them
    /// from, so a bundled CommonJS package that reads one at module-init time died
    /// with `ReferenceError: __dirname is not defined` — and, because that runs
    /// during the entry's initialization, it took the WHOLE client bundle with it
    /// (this is exactly how `next-pages-shallow-routing` failed to hydrate: Next
    /// vendors an ncc-compiled `url` polyfill that does
    /// `__nccwpck_require__.ab = __dirname + "/"`). Webpack's `target: "web"`
    /// defines the same two names per module (its `node.__dirname` "mock" default),
    /// so this is what a browser build is supposed to do.
    #[test]
    fn a_browser_bundle_defines_dirname_for_a_bundled_commonjs_module() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        // The exact shape ncc-compiled packages emit at module scope.
        fs::write(
            directory.path().join("vendored.js"),
            r#"
                const base = __dirname + "/";
                module.exports = { base, file: __filename };
            "#,
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            r#"
                const vendored = require("./vendored.js");
                console.log("base:" + vendored.base);
                console.log("file:" + vendored.file);
            "#,
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.mjs");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler
            .emit_with_options(
                &reachable,
                &output,
                EmitOptions {
                    format: ModuleFormat::BrowserEsm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();

        let code = fs::read_to_string(&output).unwrap();
        assert!(
            code.contains("const __filename=\"/index.js\",__dirname=\"/\";"),
            "the browser factory must bind the two CommonJS location ambients: {code}"
        );
        // Only the module that reads them gets the binding — the entry does not.
        assert_eq!(
            code.matches("const __filename=\"/index.js\",__dirname=\"/\";").count(),
            1,
            "the binding must be emitted per referencing module, not for every module"
        );

        // A `.mjs` file has no ambient `__dirname`, so running it proves the
        // binding is what makes the module load at all.
        let executed = node_command().arg(&output).output().unwrap();
        assert!(
            executed.status.success(),
            "{}",
            String::from_utf8_lossy(&executed.stderr)
        );
        let stdout = String::from_utf8_lossy(&executed.stdout);
        assert!(stdout.contains("base://"), "{stdout}");
        assert!(stdout.contains("file:/index.js"), "{stdout}");
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
            base: "/".to_string(),
            browser_process_shim: false,
            asset_inline_limit: 0,
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

        if node_command().arg("--version").output().is_ok() {
            let executed = node_command().arg(&output).output().unwrap();
            assert!(
                executed.status.success(),
                "{}",
                String::from_utf8_lossy(&executed.stderr)
            );
            assert_eq!(String::from_utf8_lossy(&executed.stdout), "ok\n");
        }
    }

    /// The first `/assets/...` URL with the given stem referenced by `css`.
    /// The relative `assets/<stem>-<hash>.<ext>` reference inside an emitted
    /// stylesheet (CSS asset URLs are stylesheet-relative so any public base
    /// works).
    fn asset_url_in<'c>(css: &'c str, stem: &str) -> &'c str {
        let marker = format!("url(\"assets/{stem}-");
        let start = css
            .find(&marker)
            .unwrap_or_else(|| panic!("no assets/{stem}- reference in: {css}"));
        let url = &css[start + "url(\"".len()..];
        url.split('"').next().expect("the url is terminated")
    }

    #[test]
    fn css_module_import_exports_scoped_mapping_with_vite_default_and_named_exports() {
        // Vite's default CSS Modules behavior (no `css.modules` config): the
        // default export is the locals -> scoped-names object, AND every
        // identifier-safe local is also a named export. Non-identifier locals
        // (`btn-primary`) appear only in the default object.
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("button.module.css"),
            ".btn { color: red; }\n\
             .btn:hover { color: blue; }\n\
             .btn-primary > .icon, .btn-primary::before { color: green; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import styles, { btn, icon } from './button.module.css';\n\
             console.log(styles.btn === btn, styles.icon === icon);\n\
             console.log(JSON.stringify(styles));\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();

        // The stylesheet carries the scoped selectors and no unscoped local.
        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        assert!(css.contains("._btn_"), "{css}");
        assert!(css.contains(":hover"), "{css}");
        assert!(css.contains("._btn-primary_"), "{css}");
        assert!(!css.contains(".btn "), "unscoped local leaked: {css}");
        assert!(!css.contains(".btn:"), "unscoped local leaked: {css}");
        assert!(!css.contains(".icon"), "unscoped local leaked: {css}");

        if node_command().arg("--version").output().is_ok() {
            let executed = node_command().arg(&output).output().unwrap();
            assert!(
                executed.status.success(),
                "{}",
                String::from_utf8_lossy(&executed.stderr)
            );
            let stdout = String::from_utf8_lossy(&executed.stdout);
            let mut lines = stdout.lines();
            assert_eq!(
                lines.next(),
                Some("true true"),
                "named exports must alias the default mapping: {stdout}"
            );
            let mapping: serde_json::Value =
                serde_json::from_str(lines.next().expect("mapping line")).unwrap();
            let btn = mapping["btn"].as_str().expect("btn mapping");
            assert!(
                btn.starts_with("_btn_") && btn.len() == "_btn_".len() + 8,
                "scoped name format `_btn_<hash8>`: {btn}"
            );
            let primary = mapping["btn-primary"].as_str().expect("btn-primary mapping");
            assert!(primary.starts_with("_btn-primary_"), "{primary}");
            // The scoped selector in the emitted CSS is exactly the exported name.
            assert!(css.contains(&format!(".{btn}")), "{css}");
        }
    }

    #[test]
    fn css_module_global_escape_hatch_and_same_file_composes() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("card.module.css"),
            ":global(.theme-dark) .card { color: white; }\n\
             .base { padding: 4px; }\n\
             .fancy { composes: base; color: blue; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import styles from './card.module.css';\n\
             console.log(styles.fancy);\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        bundler
            .emit(&bundler.reachable_modules_direct(), &output)
            .unwrap();

        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        // The :global(...) contents are unscoped and the wrapper is gone.
        assert!(css.contains(".theme-dark ._card_"), "{css}");
        assert!(!css.contains(":global"), "{css}");
        // composes never reaches the emitted CSS.
        assert!(!css.contains("composes"), "{css}");

        if node_command().arg("--version").output().is_ok() {
            let executed = node_command().arg(&output).output().unwrap();
            assert!(
                executed.status.success(),
                "{}",
                String::from_utf8_lossy(&executed.stderr)
            );
            let stdout = String::from_utf8_lossy(&executed.stdout);
            let names = stdout.trim().split(' ').collect::<Vec<_>>();
            assert_eq!(names.len(), 2, "self + composed: {stdout}");
            assert!(names[0].starts_with("_fancy_"), "{stdout}");
            assert!(names[1].starts_with("_base_"), "{stdout}");
            // Both classes exist in the emitted stylesheet.
            assert!(css.contains(&format!(".{}", names[0])), "{css}");
            assert!(css.contains(&format!(".{}", names[1])), "{css}");
        }
    }

    #[test]
    fn cross_file_composes_adds_a_dependency_edge_and_tracks_edits_incrementally() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let other = directory.path().join("other.module.css");
        fs::write(&other, ".bar { color: green; }").unwrap();
        fs::write(
            directory.path().join("main.module.css"),
            ".foo { composes: bar from './other.module.css'; color: red; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import styles from './main.module.css';\nconsole.log(styles.foo);\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (mut bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        // The composes source is a real graph module (the dependency edge).
        let reachable = bundler.reachable_modules_direct();
        assert!(
            reachable.iter().any(|id| id.ends_with("other.module.css")),
            "composes must create a dependency edge: {reachable:?}"
        );
        bundler.emit(&reachable, &output).unwrap();
        let first = String::from_utf8(
            node_command().arg(&output).output().unwrap().stdout,
        )
        .unwrap();
        let first_names = first.trim().split(' ').map(str::to_owned).collect::<Vec<_>>();
        assert_eq!(first_names.len(), 2, "{first}");
        assert!(first_names[0].starts_with("_foo_"), "{first}");
        assert!(first_names[1].starts_with("_bar_"), "{first}");

        // Editing the COMPOSED file re-derives through the incremental path:
        // its scoped name (content-hashed) changes, and the composer — whose
        // mapping resolves the foreign name at runtime through the module graph
        // — picks the new name up without itself being re-derived.
        fs::write(&other, ".bar { color: purple; }").unwrap();
        let update = bundler.rebuild_path(&other).unwrap();
        assert!(
            update.delta.changed.iter().any(|id| id.ends_with("other.module.css")),
            "{update:?}"
        );
        bundler.emit(&bundler.reachable_modules_direct(), &output).unwrap();
        let second = String::from_utf8(
            node_command().arg(&output).output().unwrap().stdout,
        )
        .unwrap();
        let second_names = second.trim().split(' ').map(str::to_owned).collect::<Vec<_>>();
        assert_eq!(
            first_names[0], second_names[0],
            "the composer's own scoped name is unchanged"
        );
        assert_ne!(
            first_names[1], second_names[1],
            "the composed file's scoped name must move with its content"
        );
        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        assert!(css.contains(&format!(".{}", second_names[1])), "{css}");
        assert!(css.contains("color: purple"), "{css}");
    }

    #[test]
    fn scss_global_stylesheet_compiles_through_the_css_pipeline() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("app.scss"),
            "$pad: 12px;\n#bar {\n  padding: $pad;\n  &:hover { color: red; }\n  \
             @media (min-width: 2 * 400px) { flex: 1; }\n}\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './app.scss';\nconsole.log('ok');\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        bundler
            .emit(&bundler.reachable_modules_direct(), &output)
            .unwrap();
        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        assert!(css.contains("#bar {\n  padding: 12px;\n}"), "{css}");
        assert!(css.contains("#bar:hover {\n  color: red;\n}"), "{css}");
        assert!(
            css.contains("@media (min-width: 800px) {"),
            "nested media must bubble with the evaluated prelude: {css}"
        );
    }

    #[test]
    fn scss_module_scopes_compiled_css_and_exports_the_mapping() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("_theme.scss"),
            "$clr: #e6a459;\n@mixin pulse { animation: pulse 1s infinite; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("card.module.scss"),
            "@use './theme';\n.card { color: theme.$clr; @include theme.pulse; }\n\
             @keyframes pulse { 0% { opacity: 0.5; } }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import styles from './card.module.scss';\nconsole.log(styles.card);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        bundler
            .emit(&bundler.reachable_modules_direct(), &output)
            .unwrap();
        let printed = String::from_utf8(
            node_command().arg(&output).output().unwrap().stdout,
        )
        .unwrap();
        let scoped = printed.trim();
        assert!(scoped.starts_with("_card_"), "{printed}");
        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        assert!(css.contains(&format!(".{scoped}")), "{css}");
        assert!(css.contains("color: #e6a459"), "{css}");
        // The keyframes name AND the mixin-injected animation reference are
        // scoped consistently by the CSS Modules pass.
        assert!(css.contains("@keyframes _pulse_"), "{css}");
        assert!(css.contains("animation: _pulse_"), "{css}");
    }

    #[test]
    fn editing_a_used_scss_partial_rederives_the_importing_module() {
        let directory = tempdir().unwrap();
        let partial = directory.path().join("_theme.scss");
        fs::write(&partial, "$clr: red;\n").unwrap();
        fs::write(
            directory.path().join("app.scss"),
            "@use './theme';\n.x { color: theme.$clr; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './app.scss';\nconsole.log('ok');\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (mut bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        bundler
            .emit(&bundler.reachable_modules_direct(), &output)
            .unwrap();
        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        assert!(css.contains("color: red"), "{css}");
        // The partial is not a graph module itself, but it IS a recorded css
        // source: editing it must re-derive the importing .scss module.
        assert!(bundler.is_known_module(&partial), "partial must be known");
        fs::write(&partial, "$clr: blue;\n").unwrap();
        let update = bundler.rebuild_path(&partial).unwrap();
        assert!(
            update.delta.changed.iter().any(|id| id.ends_with("app.scss")),
            "{update:?}"
        );
        bundler
            .emit(&bundler.reachable_modules_direct(), &output)
            .unwrap();
        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        assert!(css.contains("color: blue"), "{css}");
        assert!(!css.contains("color: red"), "{css}");
    }

    #[test]
    fn scss_unsupported_construct_is_a_hard_build_error() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("app.scss"),
            ".a { @extend .b; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './app.scss';\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let error = match Bundler::discover_direct(&entry) {
            Err(error) => error,
            Ok(_) => panic!("@extend must fail the build"),
        };
        assert!(
            error.contains("@extend") && error.contains("app.scss"),
            "the error must name the construct and the file: {error}"
        );
    }

    #[test]
    fn a_missing_cross_file_composes_target_throws_at_runtime_instead_of_undefined() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("other.module.css"), ".present { color: green; }")
            .unwrap();
        fs::write(
            directory.path().join("main.module.css"),
            ".foo { composes: missing from './other.module.css'; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import styles from './main.module.css';\nconsole.log(styles.foo);\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, _) = Bundler::discover_direct(&entry).unwrap();
        bundler
            .emit(&bundler.reachable_modules_direct(), &output)
            .unwrap();
        let executed = node_command().arg(&output).output().unwrap();
        assert!(
            !executed.status.success(),
            "a missing composes target must not silently yield undefined"
        );
        let stderr = String::from_utf8_lossy(&executed.stderr);
        assert!(
            stderr.contains("composes target \"missing\" is not exported by"),
            "{stderr}"
        );
    }

    #[test]
    fn css_import_statements_become_edges_with_dedup_ordering_and_media_wrap() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("base.css"), ".base { color: red; }\n").unwrap();
        fs::write(directory.path().join("cond.css"), ".cond { color: blue; }\n").unwrap();
        fs::write(
            directory.path().join("a.css"),
            "@import './base.css';\n@import './cond.css' screen;\n.a { color: black; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("b.css"),
            "@import './base.css';\n.b { color: white; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './a.css';\nimport './b.css';\nconsole.log('ok');\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (mut bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        // entry, a.css, b.css, base.css (deduped once), cond.css?media=screen.
        let reachable = bundler.reachable_modules_direct();
        assert_eq!(reachable.len(), 5, "{reachable:?}");
        assert!(
            reachable.iter().any(|id| id.ends_with("cond.css?media=screen")),
            "{reachable:?}"
        );
        bundler.emit(&reachable, &output).unwrap();

        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        assert!(!css.contains("@import"), "no unresolved @import: {css}");
        assert_eq!(
            css.matches(".base ").count(),
            1,
            "the shared import is inlined exactly once: {css}"
        );
        // Imported-before-importer ordering.
        let base = css.find(".base").unwrap();
        let a = css.find(".a ").unwrap();
        let b = css.find(".b ").unwrap();
        assert!(base < a && base < b, "{css}");
        // The media-qualified import is wrapped.
        let media = css.find("@media screen").unwrap();
        let cond = css.find(".cond").unwrap();
        let close = css[media..].find('}').unwrap() + media;
        assert!(media < cond && cond < close, "{css}");

        // Editing the media-imported file re-derives its `?media` module even
        // though the bare path is not itself a module.
        fs::write(directory.path().join("cond.css"), ".cond { color: teal; }\n").unwrap();
        let cond_path = directory.path().join("cond.css");
        assert!(bundler.is_known_module(&cond_path));
        let update = bundler.rebuild_path(&cond_path).unwrap();
        assert_eq!(update.transformed_modules, 1, "{update:?}");
        bundler.emit(&bundler.reachable_modules_direct(), &output).unwrap();
        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        assert!(css.contains("color: teal"), "{css}");
    }

    #[test]
    fn css_url_references_are_rewritten_to_hashed_assets_relative_to_each_file() {
        let directory = tempdir().unwrap();
        let sub = directory.path().join("sub");
        fs::create_dir_all(&sub).unwrap();
        fs::write(sub.join("img.png"), b"png-bytes").unwrap();
        fs::write(
            sub.join("inner.css"),
            ".inner { background: url(./img.png); }\n",
        )
        .unwrap();
        fs::write(directory.path().join("photo.jpg"), b"jpg-bytes").unwrap();
        fs::write(
            directory.path().join("top.css"),
            "@import './sub/inner.css';\n\
             .top { background: url('./photo.jpg'); }\n\
             .keep { fill: url(#gradient); background: url(data:image/gif;base64,R0); }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './top.css';\nconsole.log('ok');\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        bundler
            .emit(&bundler.reachable_modules_direct(), &output)
            .unwrap();

        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        // The nested @import's url resolved relative to THAT file (sub/img.png),
        // and both references were rewritten to hashed public URLs.
        let img_url = asset_url_in(&css, "img");
        let photo_url = asset_url_in(&css, "photo");
        assert!(img_url.ends_with(".png"), "{img_url}");
        assert!(photo_url.ends_with(".jpg"), "{photo_url}");
        assert!(!css.contains("./img.png"), "{css}");
        assert!(!css.contains("./photo.jpg"), "{css}");
        // Skipped forms survive verbatim.
        assert!(css.contains("url(#gradient)"), "{css}");
        assert!(css.contains("url(data:image/gif;base64,R0)"), "{css}");
        // The assets landed on disk with the referenced bytes.
        let assets = directory.path().join("dist/assets");
        assert_eq!(
            fs::read(assets.join(img_url.trim_start_matches("assets/"))).unwrap(),
            b"png-bytes"
        );
        assert_eq!(
            fs::read(assets.join(photo_url.trim_start_matches("assets/"))).unwrap(),
            b"jpg-bytes"
        );
    }

    #[test]
    fn a_nested_media_import_inlines_relative_urls_and_reacts_to_nested_edits() {
        let directory = tempdir().unwrap();
        let sub = directory.path().join("sub");
        fs::create_dir_all(&sub).unwrap();
        fs::write(sub.join("icon.png"), b"icon-bytes").unwrap();
        fs::write(
            sub.join("deep.css"),
            ".deep { background: url(./icon.png); }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("wrapped.css"),
            "@import './sub/deep.css';\n.wrapped { color: red; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("app.css"),
            "@import './wrapped.css' print;\n.app { color: blue; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './app.css';\nconsole.log('ok');\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (mut bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        bundler
            .emit(&bundler.reachable_modules_direct(), &output)
            .unwrap();

        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        // The whole wrapped file (including its nested import) is inside the
        // media block, and the nested file's url resolved relative to ITSELF.
        let media = css.find("@media print").unwrap();
        assert!(media < css.find(".deep").unwrap(), "{css}");
        assert!(media < css.find(".wrapped").unwrap(), "{css}");
        let icon_url = asset_url_in(&css, "icon");
        assert_eq!(
            fs::read(
                directory
                    .path()
                    .join("dist/assets")
                    .join(icon_url.trim_start_matches("assets/"))
            )
            .unwrap(),
            b"icon-bytes"
        );

        // An edit to the transitively INLINED nested file re-derives the media
        // module (tracked via css_source_files), even though neither deep.css
        // nor wrapped.css is a bare module.
        fs::write(sub.join("deep.css"), ".deep { color: orange; }\n").unwrap();
        let deep = sub.join("deep.css");
        assert!(bundler.is_known_module(&deep));
        let update = bundler.rebuild_path(&deep).unwrap();
        assert_eq!(update.transformed_modules, 1, "{update:?}");
        bundler.emit(&bundler.reachable_modules_direct(), &output).unwrap();
        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        assert!(css.contains("color: orange"), "{css}");
        let media = css.find("@media print").unwrap();
        assert!(media < css.find(".deep").unwrap(), "the edit stays wrapped: {css}");
    }

    #[test]
    fn remote_css_imports_are_hoisted_to_the_top_of_the_emitted_stylesheet() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("fonts.css"),
            "@import url(https://example.com/font.css);\n.fonts { font-family: X; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './fonts.css';\nconsole.log('ok');\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        bundler
            .emit(&bundler.reachable_modules_direct(), &output)
            .unwrap();
        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        assert!(
            css.starts_with("@import url(https://example.com/font.css);"),
            "a remote @import is only valid before all rules, so it must be hoisted: {css}"
        );
        assert!(css.contains(".fonts"), "{css}");
    }

    #[test]
    fn unsupported_css_constructs_fail_the_build_with_specific_errors() {
        // A CSS module with an at-rule the scoper cannot handle confidently.
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("odd.module.css"),
            "@tailwind base;\n.foo { color: red; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import styles from './odd.module.css';\nconsole.log(styles);\n",
        )
        .unwrap();
        let error = Bundler::discover_direct(&directory.path().join("entry.js"))
            .map(|_| ())
            .unwrap_err();
        assert!(error.contains("unsupported at-rule `@tailwind`"), "{error}");
        assert!(error.contains("odd.module.css"), "{error}");

        // An @import form we do not support.
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("x.css"), ".x{}").unwrap();
        fs::write(
            directory.path().join("layered.css"),
            "@import './x.css' layer(base);\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './layered.css';\n",
        )
        .unwrap();
        let error = Bundler::discover_direct(&directory.path().join("entry.js"))
            .map(|_| ())
            .unwrap_err();
        assert!(
            error.contains("layer(...) condition is not supported"),
            "{error}"
        );
        assert!(error.contains("layered.css"), "{error}");

        // A url() that resolves to nothing names the CSS file and the reference.
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("broken.css"),
            ".a { background: url(./missing.png); }\n",
        )
        .unwrap();
        fs::write(directory.path().join("entry.js"), "import './broken.css';\n").unwrap();
        let error = Bundler::discover_direct(&directory.path().join("entry.js"))
            .map(|_| ())
            .unwrap_err();
        assert!(error.contains("url(./missing.png)"), "{error}");
        assert!(error.contains("broken.css"), "{error}");
    }

    /// A legacy Tailwind v3 entry (`@tailwind base/components/utilities`) compiles
    /// natively through the v4 pipeline (the directives expand to the same layers), so
    /// a real v3 app is styled instead of hard-erroring as it used to.
    #[test]
    fn a_tailwind_v3_entry_compiles_natively() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("v3.css"),
            "@tailwind base;\n@tailwind components;\n@tailwind utilities;\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './v3.css';\nexport const html = '<div class=\"underline\">x</div>';\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = directory.path().join("dist/bundle.js");
        bundler
            .emit_with_options(&reachable, &output, EmitOptions::default())
            .unwrap();
        let stylesheet = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        // The `@tailwind` directives are consumed (not shipped raw) and the scanned
        // utility is generated from the vendored v4 base theme.
        assert!(!stylesheet.contains("@tailwind"), "directives must not survive: {stylesheet}");
        assert!(
            stylesheet.contains("underline") && stylesheet.contains("text-decoration"),
            "the scanned utility is generated for a v3 entry: {}",
            &stylesheet[..stylesheet.len().min(400)]
        );
    }

    /// A Tailwind v4 entry imported as a plain global stylesheet compiles
    /// through the native engine at emit time (previously a hard error that
    /// demanded `?url` — real apps, e.g. markpad, import it directly).
    #[test]
    fn a_globally_imported_tailwind_entry_is_compiled_at_emit() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("tw.css"), "@import 'tailwindcss';\n").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './tw.css';\nexport const html = '<div class=\"underline\">x</div>';\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = directory.path().join("dist/bundle.js");
        bundler
            .emit_with_options(&reachable, &output, EmitOptions::default())
            .unwrap();
        let stylesheet = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        assert!(
            !stylesheet.contains("@import 'tailwindcss'"),
            "the compiler invocation must not survive: {stylesheet}"
        );
        assert!(
            stylesheet.contains("underline") && stylesheet.contains("text-decoration"),
            "the scanned utility is generated: {}",
            &stylesheet[..stylesheet.len().min(400)]
        );
    }

    /// A Tailwind entry is the INPUT to a compiler, not a stylesheet that happens
    /// to be concatenated with its imports: `@theme`, `@utility` and plain CSS
    /// written in an `@import`ed file configure the SAME compile. Splitting the
    /// graph into separate stylesheet modules silently dropped every directive an
    /// imported file carried — an app whose design tokens live in an imported
    /// file lost its entire theme.
    #[test]
    fn an_imported_stylesheets_tailwind_directives_configure_the_entrys_compile() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("tokens.css"),
            "@theme {\n  --color-brand: #123456;\n}\n\
             @utility card-pad {\n  padding: 7px;\n}\n\
             @utility card-rule {\n  width: 3px;\n}\n\
             @property --ring-shade {\n  syntax: \"*\";\n  inherits: false;\n}\n\
             .from-tokens {\n  color: rebeccapurple;\n}\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("tw.css"),
            "@import 'tailwindcss';\n@import './tokens.css';\n\
             .card {\n  @apply text-brand card-rule;\n}\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './tw.css';\n\
             export const html = '<div class=\"card-pad from-tokens\">x</div>';\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = directory.path().join("dist/bundle.js");
        bundler
            .emit_with_options(&reachable, &output, EmitOptions::default())
            .unwrap();
        let stylesheet = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        assert!(
            !stylesheet.contains("@import"),
            "no @import may survive the compile: {stylesheet}"
        );
        // A `@theme` token from the imported file resolves an `@apply` in the entry.
        assert!(
            stylesheet.contains("#123456"),
            "the imported @theme token reaches the compile: {stylesheet}"
        );
        // An `@utility` from the imported file generates for a scanned candidate.
        assert!(
            stylesheet.contains("padding: 7px") || stylesheet.contains("padding:7px"),
            "the imported @utility generates: {stylesheet}"
        );
        // Plain rules in the imported file are emitted too.
        assert!(
            stylesheet.contains("rebeccapurple"),
            "the imported file's own rules are emitted: {stylesheet}"
        );
        // A standard block at-rule the compiler has no opinion about passes through.
        assert!(
            stylesheet.contains("@property --ring-shade"),
            "@property survives verbatim: {stylesheet}"
        );
        // And an `@apply` in the ENTRY resolves an `@utility` the IMPORTED file
        // defines — the two files are one compile, in both directions.
        assert!(
            stylesheet.contains("width: 3px") || stylesheet.contains("width:3px"),
            "the entry's @apply of an imported @utility expands: {stylesheet}"
        );
    }

    /// cal.com's exact shape: the entry itself is plain, and the `@plugin` lives in
    /// a stylesheet it `@import`s from another workspace package. The delegation
    /// gate reads the SPLICED entry, so the plugin is seen and the whole sheet is
    /// compiled by the app's own Tailwind — including a `@apply` of a utility only
    /// that plugin registers, which no native engine could answer.
    #[test]
    fn a_plugin_reached_through_an_import_delegates_the_whole_entry() {
        if Command::new("node").arg("--version").output().is_err() {
            return;
        }
        // A real Tailwind v4 install to delegate to; the corpus apps carry one.
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let modules = fs::read_dir(repo.join("integration/e2e/apps"))
            .ok()
            .into_iter()
            .flatten()
            .flatten()
            .map(|app| app.path().join("node_modules"))
            .find(|modules| modules.join("@tailwindcss/node/package.json").is_file());
        let Some(modules) = modules else {
            eprintln!("skipped: no corpus app has @tailwindcss/node installed");
            return;
        };
        let directory = tempdir().unwrap();
        let root = directory.path();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&modules, root.join("node_modules")).unwrap();
        #[cfg(windows)]
        std::os::windows::fs::symlink_dir(&modules, root.join("node_modules")).unwrap();
        fs::write(root.join("package.json"), "{\"name\":\"probe\"}\n").unwrap();
        fs::write(
            root.join("plugin.js"),
            "module.exports = function ({ addUtilities }) {\n\
               addUtilities({ '.probe-rule': { 'caret-color': 'rebeccapurple' } });\n\
             };\n",
        )
        .unwrap();
        // The plugin is declared HERE, one @import away from the entry.
        fs::write(
            root.join("tokens.css"),
            "@plugin './plugin.js';\n.from-tokens {\n  color: teal;\n}\n",
        )
        .unwrap();
        fs::write(
            root.join("tw.css"),
            "@import 'tailwindcss';\n@import './tokens.css';\n\
             .scroll-bar {\n  @apply probe-rule;\n}\n",
        )
        .unwrap();
        fs::write(
            root.join("entry.js"),
            "import './tw.css';\nexport const html = '<div class=\"flex\">x</div>';\n",
        )
        .unwrap();

        let entry = root.join("entry.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = root.join("dist/bundle.js");
        bundler
            .emit_with_options(&reachable, &output, EmitOptions::default())
            .unwrap();
        let stylesheet = fs::read_to_string(root.join("dist/bundle.css")).unwrap();
        assert!(
            stylesheet.contains("rebeccapurple"),
            "the imported file's @plugin registered the utility the entry applies: {}",
            &stylesheet[..stylesheet.len().min(600)]
        );
        assert!(
            stylesheet.contains("color: teal") || stylesheet.contains("color:teal"),
            "the imported file's own rules survive the delegated compile"
        );
        assert!(
            stylesheet.contains("display: flex") || stylesheet.contains("display:flex"),
            "diffpack's class scan still drives the delegated compile"
        );
    }

    /// `@import "some-package"` in a Tailwind entry resolves through
    /// `node_modules` with the CSS `style` condition — the resolution Tailwind
    /// itself performs, and the only way to reach a stylesheet a package
    /// publishes as `exports: { ".": { "style": "./dist/x.css" } }`.
    #[test]
    fn a_bare_css_import_resolves_through_a_packages_style_export() {
        let directory = tempdir().unwrap();
        let package = directory.path().join("node_modules/some-tokens");
        fs::create_dir_all(package.join("dist")).unwrap();
        fs::write(
            package.join("package.json"),
            "{\"name\":\"some-tokens\",\"exports\":{\".\":{\"style\":\"./dist/tokens.css\"}}}",
        )
        .unwrap();
        fs::write(
            package.join("dist/tokens.css"),
            "@utility packaged-gap {\n  gap: 11px;\n}\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("tw.css"),
            "@import 'tailwindcss';\n@import \"some-tokens\";\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './tw.css';\nexport const html = '<div class=\"packaged-gap\">x</div>';\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = directory.path().join("dist/bundle.js");
        bundler
            .emit_with_options(&reachable, &output, EmitOptions::default())
            .unwrap();
        let stylesheet = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        assert!(
            stylesheet.contains("gap: 11px") || stylesheet.contains("gap:11px"),
            "the package's published stylesheet reaches the compile: {stylesheet}"
        );
    }

    /// `@source` widens the candidate scan past the project root — how a monorepo
    /// app declares that its classes also live in sibling workspace packages —
    /// and `@source not` narrows it again. The path is anchored to the file that
    /// WROTE the directive, not to the entry that imported it.
    #[test]
    fn at_source_widens_and_at_source_not_narrows_the_candidate_scan() {
        let directory = tempdir().unwrap();
        let app = directory.path().join("app");
        fs::create_dir_all(app.join("styles")).unwrap();
        fs::create_dir_all(directory.path().join("shared/src")).unwrap();
        fs::create_dir_all(directory.path().join("shared/generated")).unwrap();
        fs::write(app.join("package.json"), "{\"name\":\"app\"}").unwrap();
        // Declared in an IMPORTED file one directory deeper, so a directive
        // anchored to the entry instead of its own file would resolve elsewhere.
        fs::write(
            app.join("styles/sources.css"),
            "@source \"../../shared/**/*.tsx\";\n\
             @source not \"../../shared/generated/**/*.tsx\";\n",
        )
        .unwrap();
        fs::write(
            app.join("tw.css"),
            "@import 'tailwindcss';\n@import './styles/sources.css';\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("shared/src/widget.tsx"),
            "export const Widget = () => <div className=\"tracking-widest\" />;\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("shared/generated/stale.tsx"),
            "export const Stale = () => <div className=\"tracking-tighter\" />;\n",
        )
        .unwrap();
        fs::write(
            app.join("entry.js"),
            "import './tw.css';\nexport const html = '<div />';\n",
        )
        .unwrap();
        let entry = app.join("entry.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = app.join("dist/bundle.js");
        bundler
            .emit_with_options(&reachable, &output, EmitOptions::default())
            .unwrap();
        let stylesheet = fs::read_to_string(app.join("dist/bundle.css")).unwrap();
        assert!(
            stylesheet.contains("tracking-widest"),
            "the @source-declared directory is scanned: {stylesheet}"
        );
        assert!(
            !stylesheet.contains("tracking-tighter"),
            "the `@source not` directory is excluded: {stylesheet}"
        );
        assert!(
            !stylesheet.contains("@source"),
            "the directive itself is consumed: {stylesheet}"
        );
    }

    #[test]
    fn source_globs_expand_braces_and_match_double_star_segments() {
        assert_eq!(
            expand_braces("a/*.{js,ts,tsx}"),
            vec!["a/*.js".to_string(), "a/*.ts".to_string(), "a/*.tsx".to_string()]
        );
        assert_eq!(expand_braces("plain/path.css"), vec!["plain/path.css".to_string()]);
        // Two groups expand as the product.
        assert_eq!(expand_braces("{a,b}/{x,y}").len(), 4);

        let pattern = |value: &str| {
            path_segments(Path::new(value))
                .into_iter()
                .collect::<Vec<_>>()
        };
        let matches = |glob: &str, path: &str| glob_matches(&pattern(glob), &pattern(path));
        assert!(matches("/a/**/*.tsx", "/a/b/c/d.tsx"));
        // `**` also matches zero segments.
        assert!(matches("/a/**/*.tsx", "/a/d.tsx"));
        assert!(!matches("/a/**/*.tsx", "/a/b/c/d.ts"));
        assert!(!matches("/a/**/*.tsx", "/other/b.tsx"));
        // `*` never crosses a segment boundary.
        assert!(!matches("/a/*.tsx", "/a/b/c.tsx"));
        assert!(matches("/a/*components*/x.tsx", "/a/my-components-here/x.tsx"));
        assert!(segment_matches("*.ts?", "main.tsx"));
        assert!(!segment_matches("*.ts?", "main.ts"));
    }

    #[test]
    fn rebuilds_only_the_changed_module_and_updates_live_reachability() {
        if node_command().arg("--version").output().is_err() {
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
        if node_command().arg("--version").output().is_err() {
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

    /// `experimentalDecorators` in the tsconfig that owns a file makes its
    /// `@decorator`s LOWER, and the `__decorate` helper the lowering calls comes
    /// from inside the binary — no `@oxc-project/runtime` install.
    ///
    /// A decorator is syntax no engine parses, so an unlowered one does not fail
    /// the build: it fails at LOAD, as `SyntaxError: Invalid or unexpected token`
    /// pointing into a minified line. Node executing the bundle (and observing the
    /// decorator's effect) is therefore the real assertion.
    #[test]
    fn legacy_decorators_lower_against_the_owning_tsconfig_and_bundle_their_helper() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("tsconfig.json"),
            r#"{"compilerOptions":{"experimentalDecorators":true}}"#,
        )
        .unwrap();
        // A method decorator that REPLACES the implementation, so the emitted
        // program is only correct if the decorator actually ran.
        fs::write(
            directory.path().join("entry.ts"),
            "function shout(_target: any, _key: string, descriptor: any) {\n\
             \x20 const inner = descriptor.value;\n\
             \x20 descriptor.value = function (...args: any[]) { return inner.apply(this, args).toUpperCase(); };\n\
             \x20 return descriptor;\n\
             }\n\
             class Greeter {\n\
             \x20 @shout\n\
             \x20 greet(name: string) { return 'hello ' + name; }\n\
             }\n\
             console.log(new Greeter().greet('world'));\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.ts");
        let output = directory.path().join("bundle.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        assert_eq!(
            reachable.len(),
            2,
            "the entry plus the embedded __decorate helper: {reachable:?}"
        );
        bundler.emit(&reachable, &output).unwrap();
        let code = fs::read_to_string(&output).unwrap();
        assert!(
            !code.contains("@oxc-project/runtime"),
            "the helper must be served from the binary, not asked of the app: {code}"
        );
        node_check(&output);
        assert_eq!(run_node(&output), "HELLO WORLD\n");
    }

    /// Without `experimentalDecorators`, a decorator is a Stage 3 decorator, which
    /// this build cannot lower. Emitting it verbatim would produce a file no engine
    /// parses, so it is a FATAL diagnostic naming the file and the decorator —
    /// never a bundle that fails at load with a SyntaxError in minified output.
    #[test]
    fn a_stage_three_decorator_is_a_fatal_diagnostic_naming_the_file_and_decorator() {
        let directory = tempdir().unwrap();
        // A tsconfig that owns the file but says nothing about decorators: the
        // TypeScript default, which is Stage 3 semantics.
        fs::write(
            directory.path().join("tsconfig.json"),
            r#"{"compilerOptions":{}}"#,
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.ts"),
            "function logged(value: any, _context: any) { return value; }\n\
             class Greeter {\n\
             \x20 @logged\n\
             \x20 greet() { return 'hi'; }\n\
             }\n\
             console.log(new Greeter().greet());\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.ts");
        let (_, update) = Bundler::discover_direct(&entry).unwrap();
        let fatal: Vec<&str> = update
            .diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.is_fatal())
            .map(|diagnostic| diagnostic.message.as_str())
            .collect();
        assert_eq!(fatal.len(), 1, "{:?}", update.diagnostics);
        assert!(fatal[0].contains("entry.ts"), "{}", fatal[0]);
        assert!(fatal[0].contains("@logged"), "{}", fatal[0]);
        assert!(
            fatal[0].contains("experimentalDecorators"),
            "the message must name the setting that would lower it: {}",
            fatal[0]
        );
    }

    /// Writes a `node_modules` JSX-runtime package whose `jsx` factory records
    /// which runtime produced an element, so a bundle can be asked, per module,
    /// what import source its JSX was lowered against.
    fn write_jsx_runtime_package(root: &Path, name: &str) {
        let package = root.join("node_modules").join(name);
        fs::create_dir_all(&package).unwrap();
        fs::write(
            package.join("package.json"),
            format!(
                r#"{{"name":"{name}","version":"1.0.0","exports":{{"./jsx-runtime":"./jsx-runtime.js"}}}}"#
            ),
        )
        .unwrap();
        fs::write(
            package.join("jsx-runtime.js"),
            format!(
                "export const Fragment = 'Fragment';\n\
                 export function jsx(tag, props) {{ return '{name}:' + tag; }}\n\
                 export const jsxs = jsx;\n"
            ),
        )
        .unwrap();
    }

    /// `compilerOptions.jsxImportSource` decides which package the automatic
    /// runtime is imported from, and it is read from the tsconfig that OWNS each
    /// file — through create-vite's solution-style root config (`{"files":[],
    /// "references":[...]}`, no `compilerOptions` at all), which a nearest-file
    /// read finds nothing in. Two files that the app's tsconfig does NOT own stay
    /// on react: a dependency's `.tsx` under `node_modules`, and diffpack's own
    /// generated `.diffpack-next/` sources, which live inside the project root and
    /// would otherwise be claimed by the app's `include`.
    #[test]
    fn jsx_import_source_comes_from_the_tsconfig_that_owns_each_file() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::create_dir_all(root.join(".diffpack-next")).unwrap();
        write_jsx_runtime_package(root, "myjsx");
        write_jsx_runtime_package(root, "react");
        // Solution-style: the root config carries no `compilerOptions` at all.
        fs::write(
            root.join("tsconfig.json"),
            r#"{"files":[],"references":[{"path":"./tsconfig.app.json"}]}"#,
        )
        .unwrap();
        // `**/*.tsx` is create-next-app's own `include`, and it reaches straight
        // into `.diffpack-next/` — which is why the guard there is not theoretical.
        fs::write(
            root.join("tsconfig.app.json"),
            r#"{"compilerOptions":{"jsx":"react-jsx","jsxImportSource":"myjsx"},
                "include":["**/*.ts","**/*.tsx"]}"#,
        )
        .unwrap();
        fs::write(
            root.join("node_modules").join("vendor.tsx"),
            "export const vendor = <span />;\n",
        )
        .unwrap();
        fs::write(
            root.join(".diffpack-next").join("generated.tsx"),
            "export const generated = <main />;\n",
        )
        .unwrap();
        let entry = root.join("src").join("entry.tsx");
        fs::write(
            &entry,
            "import { vendor } from '../node_modules/vendor.tsx';\n\
             import { generated } from '../.diffpack-next/generated.tsx';\n\
             console.log(<div />, vendor, generated);\n",
        )
        .unwrap();

        let output = root.join("bundle.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();
        assert_eq!(run_node(&output), "myjsx:div react:span react:main\n");
    }

    /// Per FILE, not per build: two sibling subtrees of ONE bundle, each with its
    /// own nearest config naming a different import source, must each be lowered
    /// against its own. A build-wide answer (first config found, or the entry's)
    /// silently hands one subtree the other's runtime, and nothing in the output
    /// says so — the JSX still compiles, it just calls into the wrong package.
    #[test]
    fn two_subtrees_with_different_nearest_configs_each_get_their_own_import_source() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let root = directory.path();
        write_jsx_runtime_package(root, "myjsx");
        write_jsx_runtime_package(root, "react");
        fs::create_dir_all(root.join("packages/preactish")).unwrap();
        fs::create_dir_all(root.join("packages/reactish")).unwrap();
        // A JS project states its options in `jsconfig.json`, a TS one in
        // `tsconfig.json`; both shapes appear in one tree here on purpose.
        fs::write(
            root.join("packages/preactish/jsconfig.json"),
            r#"{"compilerOptions":{"jsx":"react-jsx","jsxImportSource":"myjsx"}}"#,
        )
        .unwrap();
        fs::write(
            root.join("packages/reactish/tsconfig.json"),
            r#"{"compilerOptions":{"jsx":"react-jsx","jsxImportSource":"react"}}"#,
        )
        .unwrap();
        fs::write(
            root.join("packages/preactish/widget.jsx"),
            "export const widget = <span />;\n",
        )
        .unwrap();
        fs::write(
            root.join("packages/reactish/panel.tsx"),
            "export const panel = <section />;\n",
        )
        .unwrap();
        // The entry itself is under NEITHER config: it keeps oxc's react default.
        let entry = root.join("entry.jsx");
        fs::write(
            &entry,
            "import { widget } from './packages/preactish/widget.jsx';\n\
             import { panel } from './packages/reactish/panel.tsx';\n\
             console.log(<div />, widget, panel);\n",
        )
        .unwrap();

        let output = root.join("bundle.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();
        assert_eq!(run_node(&output), "react:div myjsx:span react:section\n");
    }

    /// A `.jsx` file under a tsconfig that TypeScript would NOT compile (no
    /// `allowJs`, so `include: ["src"]` does not claim it) still gets the project's
    /// import source. The bundler lowers the file whatever `tsc` would have done
    /// with it, and `preact/jsx-runtime` is the only runtime such a project has.
    #[test]
    fn a_jsx_file_gets_the_import_source_of_a_tsconfig_that_would_not_compile_it() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::create_dir_all(root.join("src")).unwrap();
        write_jsx_runtime_package(root, "myjsx");
        fs::write(
            root.join("tsconfig.json"),
            r#"{"compilerOptions":{"jsx":"react-jsx","jsxImportSource":"myjsx"},"include":["src"]}"#,
        )
        .unwrap();
        let entry = root.join("src").join("main.jsx");
        fs::write(&entry, "console.log(<div />);\n").unwrap();

        let output = root.join("bundle.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();
        assert_eq!(run_node(&output), "myjsx:div\n");
    }

    /// A JavaScript project states its compiler options in `jsconfig.json`. It is
    /// the only place such a project can put `jsxImportSource` at all, so a build
    /// that never reads it silently lowers the whole app against React.
    #[test]
    fn a_jsconfig_import_source_reaches_a_javascript_project() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::create_dir_all(root.join("src")).unwrap();
        write_jsx_runtime_package(root, "myjsx");
        fs::write(
            root.join("jsconfig.json"),
            r#"{"compilerOptions":{"jsx":"react-jsx","jsxImportSource":"myjsx"}}"#,
        )
        .unwrap();
        let entry = root.join("src").join("main.jsx");
        fs::write(&entry, "console.log(<div />);\n").unwrap();

        let output = root.join("bundle.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();
        assert_eq!(run_node(&output), "myjsx:div\n");
    }

    /// ONE app, ONE JSX runtime. `create-next-app`'s tsconfig `include`s only
    /// `**/*.ts` and `**/*.tsx`, and Next compiles JSX in `.js` (and `.mdx`) too:
    /// under a type-checking ownership rule the `.tsx` modules take the configured
    /// import source while the `.js` ones silently take React — two runtimes in one
    /// bundle, and (for a preact app) one of them not installed.
    #[test]
    fn every_extension_in_one_project_lowers_against_the_same_import_source() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::create_dir_all(root.join("app")).unwrap();
        write_jsx_runtime_package(root, "myjsx");
        write_jsx_runtime_package(root, "react");
        fs::write(
            root.join("tsconfig.json"),
            r#"{"compilerOptions":{"jsx":"preserve","jsxImportSource":"myjsx","allowJs":true},
                "include":["next-env.d.ts","**/*.ts","**/*.tsx"],
                "exclude":["node_modules"]}"#,
        )
        .unwrap();
        fs::write(
            root.join("app").join("legacy.js"),
            "export const Legacy = () => <span />;\n",
        )
        .unwrap();
        let entry = root.join("app").join("page.tsx");
        fs::write(
            &entry,
            "import { Legacy } from './legacy.js';\nconsole.log(<div />, Legacy());\n",
        )
        .unwrap();

        let output = root.join("bundle.js");
        // Next's rule: `.js` may contain JSX (`crate::parser::JsxExtensions::NextJs`).
        let config = BuildConfig {
            jsx_extensions: crate::parser::JsxExtensions::NextJs,
            ..BuildConfig::default()
        };
        let (bundler, update) = Bundler::discover_direct_with_config(&entry, &config).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();
        assert_eq!(run_node(&output), "myjsx:div myjsx:span\n");
    }

    /// A `jsx` value diffpack cannot honor names the tsconfig and the value, and
    /// stops the build — a silently mislowered module would be a bundle whose
    /// runtime import points at the wrong package.
    #[test]
    fn an_unsupported_tsconfig_jsx_value_is_a_named_hard_error() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("tsconfig.json"),
            r#"{"compilerOptions":{"jsx":"react-native-web"},"include":["src"]}"#,
        )
        .unwrap();
        let entry = root.join("src").join("entry.tsx");
        fs::write(&entry, "export const view = <div />;\n").unwrap();

        let Err(error) = Bundler::discover_direct(&entry) else {
            panic!("an unsupported tsconfig `jsx` value must stop the build");
        };
        assert!(
            error.contains("tsconfig.json")
                && error.contains("react-native-web")
                && error.contains("entry.tsx"),
            "the error must name the tsconfig, the value and the file: {error}"
        );
    }

    #[test]
    fn a_minified_chunk_runs_identically_to_its_readable_form_and_is_smaller() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.js");
        let a = directory.path().join("a.js");
        // Multi-line source with comments and whitespace, so a real whitespace/
        // syntax minification pass has something to collapse and drop.
        fs::write(
            &entry,
            concat!(
                "// entry comment\n",
                "import { a } from './a.js';\n",
                "import { b } from './b.js';\n",
                "\n",
                "function total(left, right) {\n",
                "    /* add the two operands */\n",
                "    const sum = left + right;\n",
                "    return sum;\n",
                "}\n",
                "\n",
                "console.log(total(a, b));\n",
            ),
        )
        .unwrap();
        fs::write(&a, "// module a\nexport const a = 1 + 2;\n").unwrap();
        fs::write(
            directory.path().join("b.js"),
            "// module b\nexport const b = 3;\n",
        )
        .unwrap();

        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty());
        let reachable = bundler.reachable_modules_direct();

        // Emit the readable form.
        let readable = directory.path().join("readable.js");
        bundler
            .emit_with_options(&reachable, &readable, EmitOptions::default())
            .unwrap();
        let readable_code = fs::read_to_string(&readable).unwrap();

        // Emit the minified form (same graph, `minify: true`).
        let minified = directory.path().join("minified.js");
        bundler
            .emit_with_options(
                &reachable,
                &minified,
                EmitOptions {
                    minify: true,
                    ..EmitOptions::default()
                },
            )
            .unwrap();
        let minified_code = fs::read_to_string(&minified).unwrap();

        // Behavior is identical: both run under node and print the same value.
        assert_eq!(run_node(&readable), "6\n");
        assert_eq!(
            run_node(&minified),
            run_node(&readable),
            "minified output must behave identically to the readable output"
        );

        // The minified bytes are genuinely smaller, have no comments, and are not
        // just the readable bytes passed through.
        assert!(
            minified_code.len() < readable_code.len(),
            "minified ({} bytes) must be smaller than readable ({} bytes)",
            minified_code.len(),
            readable_code.len(),
        );
        assert!(
            !minified_code.contains("entry comment")
                && !minified_code.contains("add the two operands")
                && !minified_code.contains("module a"),
            "minified output still carries comments: {minified_code}"
        );
        assert_ne!(
            minified_code, readable_code,
            "minify must actually transform the bytes"
        );
    }

    /// The build config every source-map test uses: the per-module maps the
    /// printer produces are only built when the build asks for them.
    fn source_map_config() -> BuildConfig {
        BuildConfig {
            source_maps: true,
            ..BuildConfig::default()
        }
    }

    /// A TypeScript module whose interesting identifiers sit BELOW erased,
    /// type-only statements — the exact shape a line-identity map gets wrong,
    /// because every erased line shifts the real code up by one.
    ///
    /// Returns `(source, marker_line, marker_column, call_line, call_column)`,
    /// all 0-based, for the `MARKER_ALPHA` literal and the `greet` call.
    fn typed_module_with_erased_lines() -> (&'static str, u32, u32, u32, u32) {
        // line 0: comment      line 1: interface   line 2: type alias  line 3: blank
        // line 4: export fn    line 5: const       line 6: return      line 7: }
        let source = concat!(
            "// a leading comment\n",
            "interface Props { label: string }\n",
            "type Unused = number\n",
            "\n",
            "export function greet(props: Props) {\n",
            "  const marker = \"MARKER_ALPHA\"\n",
            "  return props.label + marker + globalThis.who\n",
            "}\n",
        );
        (source, 5, 17, 4, 16)
    }

    /// Finds the 0-based (line, column) of `needle` in `text`, in UTF-16 columns.
    fn position_of(text: &str, needle: &str) -> (u32, u32) {
        let byte = text
            .find(needle)
            .unwrap_or_else(|| panic!("`{needle}` must be present in:\n{text}"));
        let prefix = &text[..byte];
        let line_start = prefix.rfind('\n').map_or(0, |newline| newline + 1);
        (
            prefix.matches('\n').count() as u32,
            crate::source_map::utf16_len(&text[line_start..byte]),
        )
    }

    /// A READABLE chunk's map must resolve a known identifier to the EXACT
    /// original line AND column it came from — not to the generated line's index,
    /// which is what a line-identity guess produces and what every erased
    /// TypeScript line above the identifier makes wrong.
    #[test]
    fn a_readable_chunk_map_resolves_a_known_identifier_to_its_exact_original_line_and_column() {
        use oxc_sourcemap::SourceMap;

        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.ts");
        let a = directory.path().join("a.ts");
        let (source, marker_line, marker_column, greet_line, greet_column) =
            typed_module_with_erased_lines();
        fs::write(&a, source).unwrap();
        fs::write(
            &entry,
            "import { greet } from './a.ts';\nconsole.log(greet({ label: \"x\" }));\n",
        )
        .unwrap();

        let (bundler, update) =
            Bundler::discover_direct_with_config(&entry, &source_map_config()).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = directory.path().join("out.js");
        bundler
            .emit_with_options(
                &reachable,
                &output,
                EmitOptions {
                    source_map: true,
                    ..EmitOptions::default()
                },
            )
            .unwrap();

        let code = fs::read_to_string(&output).unwrap();
        let map_json = fs::read_to_string(directory.path().join("out.js.map")).unwrap();
        let map = SourceMap::from_json_string(&map_json).unwrap();
        let table = map.generate_lookup_table();

        // The string literal in the EMITTED chunk resolves to the literal in a.ts:
        // line 5, column 17 — four lines below where a line-identity map would put
        // it, because the comment, the interface and the type alias all vanished.
        let (line, column) = position_of(&code, "\"MARKER_ALPHA\"");
        let token = map
            .lookup_token(&table, line, column)
            .expect("the literal's position must be mapped");
        assert_eq!(
            (
                token.get_source_id().and_then(|id| map.get_source(id)),
                token.get_src_line(),
                token.get_src_col(),
            ),
            (Some("diffpack:///a.ts"), marker_line, marker_column),
            "the emitted literal must resolve to a.ts {}:{marker_column}, got {:?}",
            marker_line + 1,
            (token.get_src_line(), token.get_src_col()),
        );

        // ...and so does the function NAME, at its own exact column.
        let (line, column) = position_of(&code, "greet(props)");
        let token = map
            .lookup_token(&table, line, column)
            .expect("the declaration's position must be mapped");
        assert_eq!(
            (token.get_src_line(), token.get_src_col()),
            (greet_line, greet_column),
            "the emitted `greet` declaration must resolve to a.ts {}:{greet_column}",
            greet_line + 1,
        );

        // Every mapped column must be a REAL column: a map that had given up and
        // pinned everything to column 0 would pass a line-only check.
        assert!(
            map.get_tokens().any(|token| token.get_src_col() > 0),
            "the map must carry real columns, not column 0 for everything"
        );

        // A bundler-synthesized line owns no original position and must be
        // EXPLICITLY unmapped rather than be attributed to whatever module is
        // nearby. Explicitly matters: omitting a token does not mark a line
        // unmapped, because a consumer resolves a position to the last mapping at
        // or before it anywhere in the file, so a line with nothing on it inherits
        // the previous line's origin.
        let (line, _) = position_of(&code, "console.log");
        let separator = line - 1;
        let marker = map
            .get_tokens()
            .find(|token| token.get_dst_line() == separator)
            .expect("the blank separator line must carry an explicit unmapped marker");
        assert_eq!(
            (marker.get_dst_col(), marker.get_source_id()),
            (0, None),
            "the marker must be a source-less segment at the start of the line, got {marker:?}"
        );
        assert!(
            map.lookup_token(&table, separator, 0)
                .is_none_or(|token| token.get_source_id().is_none()),
            "resolving the blank separator line must not name any original source"
        );
    }

    /// Every generated line of a readable chunk that no module accounts for must
    /// say so, with its own unmapped segment.
    ///
    /// This is the whole honesty mechanism, and leaving the token out does NOT
    /// achieve it: Node's `--enable-source-maps` (which is how `diffpack start`
    /// runs a server) and DevTools both binary-search the flattened mapping list
    /// and return the last entry at or before the queried position, IGNORING line
    /// boundaries. So a bundler-authored line with no segments resolves to
    /// whatever author code was mapped before it — which is how a frame inside the
    /// bundler's own `__require` came out attributed to a component, at a line and
    /// column that exist, in a file that has no such code.
    #[test]
    fn every_line_a_module_does_not_account_for_carries_an_explicit_unmapped_marker() {
        use oxc_sourcemap::SourceMap;

        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.ts");
        let a = directory.path().join("a.ts");
        let (source, ..) = typed_module_with_erased_lines();
        fs::write(&a, source).unwrap();
        // A CommonJS dependency forces the full registry runtime into the chunk,
        // so the chunk really does interleave author code with bundler-authored
        // text — which is the situation the markers exist for.
        fs::write(
            directory.path().join("legacy.cjs"),
            "module.exports = { legacy: 1 };\n",
        )
        .unwrap();
        fs::write(
            &entry,
            "import { greet } from \"./a\";\nimport legacy from \"./legacy.cjs\";\n\
             console.log(greet({ label: \"x\" }), legacy);\n",
        )
        .unwrap();

        let (bundler, update) =
            Bundler::discover_direct_with_config(&entry, &source_map_config()).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = directory.path().join("out.js");
        bundler
            .emit_with_options(
                &reachable,
                &output,
                EmitOptions {
                    source_map: true,
                    ..EmitOptions::default()
                },
            )
            .unwrap();

        let code = fs::read_to_string(&output).unwrap();
        let map_json = fs::read_to_string(directory.path().join("out.js.map")).unwrap();
        let map = SourceMap::from_json_string(&map_json).unwrap();
        let mut mapped_lines: HashSet<u32> = HashSet::new();
        let mut marked_lines: HashSet<u32> = HashSet::new();
        for token in map.get_tokens() {
            match token.get_source_id() {
                Some(_) => mapped_lines.insert(token.get_dst_line()),
                None => marked_lines.insert(token.get_dst_line()),
            };
        }
        let total = line_count(&code);
        let unaccounted: Vec<u32> = (0..total)
            .filter(|line| !mapped_lines.contains(line) && !marked_lines.contains(line))
            .collect();
        assert!(
            unaccounted.is_empty(),
            "generated lines {unaccounted:?} of a {total}-line chunk carry neither a mapping \
             nor an unmapped marker, so a consumer resolves them to the last mapping before \
             them:\n{code}"
        );
        assert!(
            !marked_lines.is_empty() && !mapped_lines.is_empty(),
            "the chunk must have both kinds of line — runtime/glue and module code — for this \
             to be testing anything (mapped: {}, marked: {})",
            mapped_lines.len(),
            marked_lines.len()
        );
        // The runtime's own `__require` is bundler-authored: resolving a position
        // in it must name no source at all.
        let table = map.generate_lookup_table();
        let (throw_line, throw_column) = position_of(&code, "Module is not loaded");
        assert!(
            map.lookup_token(&table, throw_line, throw_column)
                .is_none_or(|token| token.get_source_id().is_none()),
            "a position inside the bundler's own runtime must resolve to no original source"
        );
    }

    /// A `sources` label is a module's IDENTITY — DevTools' source tree and every
    /// error reporter dedupe on it — so it must carry the module's directory and
    /// stay the same in every chunk it appears in.
    ///
    /// The failure this locks out: a root computed per MAP collapses to the
    /// module's own directory whenever a chunk holds one module, and the label
    /// becomes a bare file name. On cal.com that turned nine different
    /// `pages/setup/index.tsx` files into one `diffpack:///Setup.tsx`, and thirty
    /// different `add.ts` files into one `diffpack:///add.ts`.
    #[test]
    fn same_named_modules_in_different_chunks_keep_distinct_directory_qualified_labels() {
        use oxc_sourcemap::SourceMap;

        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::write(root.join("package.json"), r#"{"name":"labels"}"#).unwrap();
        for area in ["alpha", "beta"] {
            fs::create_dir_all(root.join("src").join(area)).unwrap();
            fs::write(
                root.join("src").join(area).join("Setup.ts"),
                format!("export const AREA = \"{area}\";\n"),
            )
            .unwrap();
        }
        let entry = root.join("src").join("entry.ts");
        // Dynamic imports put each `Setup.ts` in its own chunk, which is exactly
        // when a per-chunk root degenerates to a bare file name.
        fs::write(
            &entry,
            "const both = [import(\"./alpha/Setup\"), import(\"./beta/Setup\")];\n\
             Promise.all(both).then((loaded) => console.log(loaded.length));\n",
        )
        .unwrap();

        let (bundler, update) =
            Bundler::discover_direct_with_config(&entry, &source_map_config()).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = root.join("out").join("bundle.js");
        bundler
            .emit_with_options(
                &reachable,
                &output,
                EmitOptions {
                    source_map: true,
                    ..EmitOptions::default()
                },
            )
            .unwrap();

        let mut labels: Vec<String> = Vec::new();
        for file in fs::read_dir(root.join("out")).unwrap() {
            let path = file.unwrap().path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("map") {
                continue;
            }
            let json = fs::read_to_string(&path).unwrap();
            let map = SourceMap::from_json_string(&json).unwrap();
            labels.extend(map.get_sources().map(str::to_owned));
        }
        assert!(
            labels.contains(&"diffpack:///src/alpha/Setup.ts".to_string())
                && labels.contains(&"diffpack:///src/beta/Setup.ts".to_string()),
            "each module must be named by its path from the project root, got {labels:?}"
        );
    }

    /// A module OUTSIDE the project root (a package in a store elsewhere, a
    /// symlinked workspace, another volume) must never publish its absolute path:
    /// that names the machine and the user, and production maps are served to
    /// browsers. It must still be told apart from any other file, so the label
    /// keeps the path within its own package and disambiguates it.
    #[test]
    fn a_module_outside_the_project_root_is_labelled_without_leaking_its_absolute_path() {
        use oxc_sourcemap::SourceMap;

        let directory = tempdir().unwrap();
        let root = directory.path();
        let project = root.join("project");
        fs::create_dir_all(project.join("src")).unwrap();
        fs::write(project.join("package.json"), r#"{"name":"project"}"#).unwrap();
        let outside = root.join("elsewhere").join("pkg");
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("package.json"), r#"{"name":"faraway"}"#).unwrap();
        fs::write(
            outside.join("index.js"),
            "export const FAR = \"far\";\nexport function far(x) { return x + FAR; }\n",
        )
        .unwrap();
        let entry = project.join("src").join("entry.js");
        fs::write(
            &entry,
            "import { far } from \"../../elsewhere/pkg/index.js\";\nconsole.log(far(1));\n",
        )
        .unwrap();

        let (bundler, update) =
            Bundler::discover_direct_with_config(&entry, &source_map_config()).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = project.join("out.js");
        bundler
            .emit_with_options(
                &reachable,
                &output,
                EmitOptions {
                    source_map: true,
                    ..EmitOptions::default()
                },
            )
            .unwrap();

        let map_json = fs::read_to_string(project.join("out.js.map")).unwrap();
        let map = SourceMap::from_json_string(&map_json).unwrap();
        let sources = map.get_sources().collect::<Vec<_>>();
        let leaked = root.to_string_lossy().to_string();
        assert!(
            sources.iter().all(|source| !source.contains(&leaked)
                && !source.contains("..")
                && source.starts_with("diffpack:///")),
            "no label may carry an absolute path or a traversal, got {sources:?} (root {leaked})"
        );
        assert!(
            sources
                .iter()
                .any(|source| source.contains("external/") && source.ends_with("pkg/index.js")),
            "the outside module must still be identifiable, got {sources:?}"
        );
        assert!(
            sources.contains(&"diffpack:///src/entry.js"),
            "a module INSIDE the project keeps its project-relative path, got {sources:?}"
        );
    }

    /// A module whose source diffpack REWROTE before parsing must not be
    /// presented as the file on disk: the map's positions index the rewritten
    /// text, so the label says so and the inlined content is that text.
    #[test]
    fn a_rewritten_source_is_labelled_and_carries_the_text_its_positions_index() {
        use oxc_sourcemap::SourceMap;

        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.ts");
        // A Vite `define` rewrites the SOURCE before it is parsed, so every span
        // below the substitution is measured against text that is not on disk.
        fs::write(
            &entry,
            "const flag = __BUILD_FLAG__\nconsole.log(flag, globalThis.who)\n",
        )
        .unwrap();

        let config = BuildConfig {
            source_maps: true,
            defines: vec![("__BUILD_FLAG__".to_string(), "\"enabled\"".to_string())],
            ..BuildConfig::default()
        };
        let (bundler, update) = Bundler::discover_direct_with_config(&entry, &config).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = directory.path().join("out.js");
        bundler
            .emit_with_options(
                &reachable,
                &output,
                EmitOptions {
                    source_map: true,
                    ..EmitOptions::default()
                },
            )
            .unwrap();

        let map_json = fs::read_to_string(directory.path().join("out.js.map")).unwrap();
        let map = SourceMap::from_json_string(&map_json).unwrap();
        let sources = map.get_sources().collect::<Vec<_>>();
        assert_eq!(
            sources,
            vec!["diffpack:///entry.ts?diffpack-generated=vite-replace&diffpack-graph=server"],
            "a rewritten source must be labelled as generated (and by which graph generated \
             it, since the same file rewrites differently per graph), not as the file on disk"
        );
        let content = map.get_source_content(0).expect("content must be inlined");
        assert!(
            content.contains("\"enabled\"") && !content.contains("__BUILD_FLAG__"),
            "sourcesContent must be the REWRITTEN text the positions were measured \
             against, got: {content}"
        );
    }

    /// DEV: the Fast Refresh instrumentation edits a module's lowered code AFTER
    /// the printer measured it — a whole line inserted at the top, and
    /// `import.meta.hot` rewritten in place. The map must move with it, or every
    /// position in a dev build is one line off.
    #[test]
    fn the_fast_refresh_instrumentation_moves_the_map_with_the_code_it_edits() {
        use oxc_sourcemap::SourceMap;

        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.tsx");
        // A component (so the Fast Refresh preamble is injected) that also reads
        // `import.meta.hot` (so the in-place rewrite runs too).
        let source = concat!(
            "// a comment\n",
            "type Props = { label: string }\n",
            "export function Widget(props: Props) {\n",
            "  const marker = \"MARKER_ALPHA\"\n",
            "  return marker + props.label\n",
            "}\n",
            "if (import.meta.hot) { import.meta.hot.accept() }\n",
            "console.log(Widget({ label: globalThis.who }))\n",
        );
        fs::write(&entry, source).unwrap();

        let config = BuildConfig {
            source_maps: true,
            hmr: true,
            target: Target::Client,
            ..BuildConfig::default()
        };
        let (bundler, update) = Bundler::discover_direct_with_config(&entry, &config).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = directory.path().join("out.js");
        bundler
            .emit_with_options(
                &reachable,
                &output,
                EmitOptions {
                    source_map: true,
                    hmr: true,
                    format: ModuleFormat::BrowserEsm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();

        let code = fs::read_to_string(&output).unwrap();
        assert!(
            code.contains("$RefreshReg$"),
            "the module must have been instrumented, or this test proves nothing"
        );
        let map_json = fs::read_to_string(directory.path().join("out.js.map")).unwrap();
        let map = SourceMap::from_json_string(&map_json).unwrap();
        let table = map.generate_lookup_table();

        let (expected_line, expected_column) = position_of(source, "\"MARKER_ALPHA\"");
        let (line, column) = position_of(&code, "\"MARKER_ALPHA\"");
        let token = map
            .lookup_token(&table, line, column)
            .expect("the literal must still be mapped after instrumentation");
        assert_eq!(
            (token.get_src_line(), token.get_src_col()),
            (expected_line, expected_column),
            "the instrumented module's map must still point at the original literal",
        );
    }

    /// Emitting a map from a bundler that was never asked to build the per-module
    /// maps is refused, loudly. There is no cheaper, guessed map to fall back to.
    #[test]
    fn emitting_a_source_map_without_the_per_module_maps_is_a_hard_error() {
        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.js");
        fs::write(&entry, "console.log(globalThis.who);\n").unwrap();
        let (bundler, _) = Bundler::discover_direct(&entry).unwrap();
        let reachable = bundler.reachable_modules_direct();
        let error = bundler
            .emit_with_options(
                &reachable,
                &directory.path().join("out.js"),
                EmitOptions {
                    source_map: true,
                    ..EmitOptions::default()
                },
            )
            .expect_err("a map with no measured positions must be refused");
        assert!(
            error.contains("source_maps"),
            "the refusal must name the setting that fixes it, got: {error}"
        );
    }

    /// A multi-chunk app whose modules are split across chunks by dynamic import,
    /// so the coverage assertions below run over MORE than the entry chunk.
    fn code_split_source_map_project(directory: &Path) -> PathBuf {
        let entry = directory.join("entry.ts");
        fs::write(
            &entry,
            "import { shared } from \"./shared\";\n\
             export async function boot(): Promise<string> {\n\
             \x20 const lazy = await import(\"./lazy\");\n\
             \x20 return shared() + lazy.lazily();\n\
             }\n",
        )
        .unwrap();
        fs::write(
            directory.join("shared.ts"),
            "interface Erased { gone: boolean }\n\
             type AlsoErased = string;\n\
             export function shared(): string {\n\
             \x20 return \"SHARED_MARKER\";\n\
             }\n",
        )
        .unwrap();
        fs::write(
            directory.join("lazy.ts"),
            "import { shared } from \"./shared\";\n\
             type Gone = number;\n\
             export function lazily(): string {\n\
             \x20 return \"LAZY_MARKER\" + shared();\n\
             }\n",
        )
        .unwrap();
        entry
    }

    /// Every JS file an emit writes either carries NO `sourceMappingURL` at all or
    /// carries one whose file was really written, and whose `file` field names the
    /// chunk it belongs to.
    ///
    /// A dangling `sourceMappingURL` is not a cosmetic defect: the browser fetches
    /// it on every load of the chunk and logs a failure, and a `file` field naming
    /// some other chunk sends a map consumer to the wrong bytes. Both are the kind
    /// of drift that appears the moment a second writer (here: the dev HMR
    /// micro-chunk) names its sidecar itself, which is why the naming lives in one
    /// place — [`Bundler::source_map_sidecar`].
    #[test]
    fn every_emitted_chunk_points_at_a_map_that_exists_and_names_itself() {
        for minify in [false, true] {
            let directory = tempdir().unwrap();
            let entry = code_split_source_map_project(directory.path());
            let (bundler, update) =
                Bundler::discover_direct_with_config(&entry, &source_map_config()).unwrap();
            assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
            let reachable = bundler.reachable_modules_direct();
            let out_root = directory.path().join("out");
            bundler
                .emit_public(
                    &reachable,
                    &out_root,
                    EmitOptions {
                        source_map: true,
                        minify,
                        ..EmitOptions::default()
                    },
                )
                .unwrap();

            let public = out_root.join("public");
            let mut checked = 0;
            for entry in fs::read_dir(&public).unwrap() {
                let path = entry.unwrap().path();
                let name = path.file_name().unwrap().to_str().unwrap().to_string();
                if !name.ends_with(".js") {
                    continue;
                }
                checked += 1;
                let code = fs::read_to_string(&path).unwrap();
                let reference = code
                    .rsplit("//# sourceMappingURL=")
                    .next()
                    .filter(|_| code.contains("//# sourceMappingURL="))
                    .map(|tail| tail.trim().to_string())
                    .unwrap_or_else(|| {
                        panic!("{name} was emitted with source maps on but names no map")
                    });
                let map_path = public.join(&reference);
                assert!(
                    map_path.is_file(),
                    "{name} points at {reference}, which was never written — a browser \
                     fetches that on every load and gets a 404"
                );
                let map: serde_json::Value =
                    serde_json::from_str(&fs::read_to_string(&map_path).unwrap()).unwrap();
                assert_eq!(
                    map.get("file").and_then(|value| value.as_str()),
                    Some(name.as_str()),
                    "{reference} claims to describe a different chunk"
                );
            }
            assert!(
                checked > 1,
                "the fixture must emit MORE than one chunk (minify={minify}), or this \
                 proves nothing about chunks past the entry"
            );
        }
    }

    /// The dev HMR micro-chunk — the code the developer is editing RIGHT NOW — ships
    /// with its own map, and that map resolves back to the edited file.
    ///
    /// This is the most user-visible source map diffpack writes: it is the one a
    /// stack trace lands in seconds after a save. It previously shipped with none at
    /// all, so the hot-updated module was the one region of a dev session with no
    /// mapping.
    #[test]
    fn the_hmr_micro_chunk_ships_a_map_that_resolves_to_the_edited_source() {
        use oxc_sourcemap::SourceMap;

        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.ts");
        let edited = directory.path().join("edited.ts");
        fs::write(&entry, "import { hot } from \"./edited\";\nconsole.log(hot());\n").unwrap();
        // Type-only lines above the marker, so a line-identity guess would be wrong.
        let source = "interface Erased { gone: boolean }\n\
                      type AlsoErased = string;\n\
                      \n\
                      export function hot(): string {\n\
                      \x20 return \"HOT_MARKER\";\n\
                      }\n";
        fs::write(&edited, source).unwrap();

        let config = BuildConfig {
            source_maps: true,
            hmr: true,
            target: Target::Client,
            ..BuildConfig::default()
        };
        let (bundler, update) = Bundler::discover_direct_with_config(&entry, &config).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let changed: BTreeSet<ModuleId> = reachable
            .iter()
            .filter(|id| id.contains("edited.ts"))
            .cloned()
            .collect();
        assert_eq!(changed.len(), 1, "the edited module must be in the graph");

        let chunk = directory.path().join("client.hmr.js");
        let wrote = bundler
            .write_hmr_chunk(
                &reachable,
                &changed,
                "client.js",
                EmitOptions {
                    source_map: true,
                    hmr: true,
                    format: ModuleFormat::BrowserEsm,
                    ..EmitOptions::default()
                },
                ModuleFormat::BrowserEsm,
                &chunk,
            )
            .unwrap();
        assert!(wrote, "the edited module is live, so a micro-chunk must render");

        let code = fs::read_to_string(&chunk).unwrap();
        assert!(
            code.trim_end().ends_with("//# sourceMappingURL=client.hmr.js.map"),
            "the micro-chunk must name its map, or the browser never loads it"
        );
        let map_path = directory.path().join("client.hmr.js.map");
        let map_json = fs::read_to_string(&map_path).unwrap();
        let map = SourceMap::from_json_string(&map_json).unwrap();
        let table = map.generate_lookup_table();

        let (expected_line, expected_column) = position_of(source, "\"HOT_MARKER\"");
        let (line, column) = position_of(&code, "\"HOT_MARKER\"");
        let token = map
            .lookup_token(&table, line, column)
            .expect("the marker must be mapped in the micro-chunk");
        assert_eq!(
            (token.get_src_line(), token.get_src_col()),
            (expected_line, expected_column),
            "the micro-chunk's map must resolve to the edited file's real position, \
             not to the generated line number"
        );
    }

    /// With source maps OFF, the micro-chunk carries no dangling reference: no map
    /// file, and no `sourceMappingURL` for the browser to chase.
    #[test]
    fn the_hmr_micro_chunk_names_no_map_when_source_maps_are_off() {
        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.js");
        let edited = directory.path().join("edited.js");
        fs::write(&entry, "import { hot } from \"./edited\";\nconsole.log(hot());\n").unwrap();
        fs::write(&edited, "export function hot() {\n  return \"HOT\";\n}\n").unwrap();
        let config = BuildConfig {
            hmr: true,
            target: Target::Client,
            ..BuildConfig::default()
        };
        let (bundler, _) = Bundler::discover_direct_with_config(&entry, &config).unwrap();
        let reachable = bundler.reachable_modules_direct();
        let changed: BTreeSet<ModuleId> = reachable
            .iter()
            .filter(|id| id.contains("edited.js"))
            .cloned()
            .collect();
        let chunk = directory.path().join("client.hmr.js");
        assert!(
            bundler
                .write_hmr_chunk(
                    &reachable,
                    &changed,
                    "client.js",
                    EmitOptions {
                        source_map: false,
                        hmr: true,
                        format: ModuleFormat::BrowserEsm,
                        ..EmitOptions::default()
                    },
                    ModuleFormat::BrowserEsm,
                    &chunk,
                )
                .unwrap()
        );
        assert!(!fs::read_to_string(&chunk).unwrap().contains("sourceMappingURL"));
        assert!(!directory.path().join("client.hmr.js.map").exists());
    }

    /// `__dirname`/`__filename` in a SPLIT Node ESM build. Every chunk is its own ES
    /// module, so it does NOT close over the entry's bindings: a bundled CommonJS module
    /// that reads `__dirname` and lands behind a dynamic `import()` threw
    /// `ReferenceError: __dirname is not defined in ES module scope` the moment its chunk
    /// was loaded. That is not hypothetical — it is how cal.com's `pages/api/**` routes
    /// died: Prisma's generated client reads `__dirname`, and in the SSR graph it is
    /// reachable ONLY through those lazily-imported route chunks. The prelude therefore
    /// belongs on every Node ESM chunk, not just the entry.
    #[test]
    fn a_split_node_chunk_defines_dirname_for_a_bundled_commonjs_module() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        // The exact shape ncc-compiled / generated CJS packages emit at module scope.
        fs::write(
            directory.path().join("vendored.js"),
            "const base = __dirname + \"/\";\nmodule.exports = { base, file: __filename };\n",
        )
        .unwrap();
        // `lazy.js` is only reachable through the dynamic import, so it (and the CJS
        // module it pulls in) lands in a chunk of its own.
        fs::write(
            directory.path().join("lazy.js"),
            "const vendored = require(\"./vendored.js\");\nexport const where = vendored.base;\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import(\"./lazy.js\").then(({ where }) => console.log(\"base:\" + where));\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output_root = directory.path().join(".diffpack-output");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler
            .emit_server(&reachable, &output_root, EmitOptions::default())
            .unwrap();

        let server_dir = output_root.join("server");
        let chunk = server_dir.join("server.chunk-1.mjs");
        assert!(chunk.is_file(), "the dynamic import lands in its own chunk");
        assert!(
            fs::read_to_string(&chunk)
                .unwrap()
                .contains("const __dirname = __diffpackDirname(__filename)"),
            "the split chunk must define __dirname from its own import.meta.url",
        );

        // Running it is the real proof: the chunk is imported at runtime, and without
        // the prelude that import rejects with a ReferenceError.
        let executed = node_command().arg(server_dir.join("server.mjs")).output().unwrap();
        assert!(
            executed.status.success(),
            "{}",
            String::from_utf8_lossy(&executed.stderr)
        );
        let stdout = String::from_utf8_lossy(&executed.stdout);
        // Compared against the CANONICAL path: macOS resolves the temp dir through
        // `/private`, and `import.meta.url` carries the resolved form.
        let canonical = server_dir.canonicalize().unwrap();
        assert!(
            stdout.contains(&format!("base:{}/", canonical.display())),
            "the chunk resolves __dirname to its own directory: {stdout}"
        );
    }

    /// A hot-updated module must land in the SAME environment its graph was emitted
    /// for. `__dirname`/`__filename` are the sharp edge: browser output substitutes
    /// the stubs `"/index.js"` and `"/"` (a browser has no CommonJS locations), while
    /// Node ESM output binds the entry's real values. Rendering a SERVER micro-chunk as
    /// browser output therefore swaps a server module's file paths for stubs the
    /// instant it is hot-updated — a fault that is invisible until an edit, and then
    /// only on a module that reads a file. `write_hmr_chunk` takes the format
    /// explicitly for this reason; this pins both sides of the choice.
    #[test]
    fn the_hmr_micro_chunk_renders_dirname_for_the_format_its_graph_was_emitted_for() {
        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.js");
        let edited = directory.path().join("edited.js");
        fs::write(&entry, "import { where } from \"./edited\";\nconsole.log(where());\n").unwrap();
        fs::write(&edited, "export function where() {\n  return __dirname;\n}\n").unwrap();
        let config = BuildConfig {
            hmr: true,
            ..BuildConfig::default()
        };
        let (bundler, _) = Bundler::discover_direct_with_config(&entry, &config).unwrap();
        let reachable = bundler.reachable_modules_direct();
        let changed: BTreeSet<ModuleId> = reachable
            .iter()
            .filter(|id| id.contains("edited.js"))
            .cloned()
            .collect();
        let options = EmitOptions {
            hmr: true,
            ..EmitOptions::default()
        };

        let browser = directory.path().join("client.hmr.js");
        assert!(
            bundler
                .write_hmr_chunk(
                    &reachable,
                    &changed,
                    "client.js",
                    options,
                    ModuleFormat::BrowserEsm,
                    &browser,
                )
                .unwrap()
        );
        assert!(
            fs::read_to_string(&browser).unwrap().contains("__dirname=\"/\""),
            "a browser micro-chunk must carry the browser CommonJS-location stubs",
        );

        let node = directory.path().join("server.hmr.mjs");
        assert!(
            bundler
                .write_hmr_chunk(
                    &reachable,
                    &changed,
                    "server.mjs",
                    options,
                    ModuleFormat::Esm,
                    &node,
                )
                .unwrap()
        );
        assert!(
            !fs::read_to_string(&node).unwrap().contains("__dirname=\"/\""),
            "a Node micro-chunk must NOT stub __dirname; it binds the entry's real value",
        );
    }

    #[test]
    fn a_minified_chunk_emits_a_composed_source_map_resolving_to_the_original_source() {
        use oxc_sourcemap::SourceMap;

        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.ts");
        let a = directory.path().join("a.ts");
        // `a.ts` must still contribute generated bytes AFTER compression, or there
        // is no cross-module position left to sample. A single `const` would be
        // inlined into its one use and the module would vanish entirely (correctly
        // — that is what esbuild does too), so `a.ts` exports a function called
        // from two places, which the minifier keeps as a real binding. Its
        // interesting lines sit below erased TypeScript, so a line-identity map
        // resolves them four lines too high.
        let (source, marker_line, marker_column, greet_line, greet_column) =
            typed_module_with_erased_lines();
        fs::write(&a, source).unwrap();
        fs::write(
            &entry,
            "import { greet } from './a.ts';\nconsole.log(greet({ label: globalThis.who }));\nconsole.log(greet({ label: globalThis.other }));\n",
        )
        .unwrap();

        let (bundler, update) =
            Bundler::discover_direct_with_config(&entry, &source_map_config()).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();

        let output = directory.path().join("out.js");
        bundler
            .emit_with_options(
                &reachable,
                &output,
                EmitOptions {
                    minify: true,
                    source_map: true,
                    ..EmitOptions::default()
                },
            )
            .unwrap();

        // The emitted (minified) chunk references its sibling map.
        let code = fs::read_to_string(&output).unwrap();
        assert!(
            code.contains("//# sourceMappingURL=out.js.map"),
            "minified chunk must reference its sibling map: {code}"
        );
        // It is genuinely minified (no source comments/newlines-per-statement).
        assert!(
            !code.contains("MARKER_ALPHA\";\n"),
            "the chunk must be minified, got: {code}"
        );

        // The map is valid JSON listing the real original sources with their
        // content inlined, under project-relative, traversal-free labels.
        let map_path = directory.path().join("out.js.map");
        let map_json = fs::read_to_string(&map_path).unwrap();
        let map = SourceMap::from_json_string(&map_json).unwrap();
        let sources = map.get_sources().collect::<Vec<_>>();
        assert!(
            sources.iter().any(|source| source.ends_with("a.ts"))
                && sources.iter().any(|source| source.ends_with("entry.ts")),
            "sources must list the real original modules, got {sources:?}"
        );
        assert!(
            sources
                .iter()
                .all(|source| source.starts_with("diffpack:///") && !source.contains("..")),
            "source labels must be project-relative and traversal-free, got {sources:?}"
        );
        let a_index = sources
            .iter()
            .position(|source| source.ends_with("a.ts"))
            .expect("a.ts must be a source");
        let a_content = map.get_source_content(a_index as u32);
        assert!(
            a_content.is_some_and(|content| content.contains("MARKER_ALPHA")),
            "sourcesContent must carry the real a.ts source, got {a_content:?}"
        );

        let table = map.generate_lookup_table();
        // A sampled MINIFIED position — the string literal that came from a.ts —
        // decodes back to a.ts at an EXACT line and column. The minifier inlined
        // the `marker` constant into its use, so the honest answer is the USE
        // site, not the declaration: line 7, column 23 of a.ts. Under the
        // line-identity map every position on the minified chunk's single line
        // resolved to line 1 of whichever module owned readable line 0.
        let (inlined_line, inlined_column) = position_of(source, "marker + globalThis");
        assert_ne!(
            (inlined_line, inlined_column),
            (marker_line, marker_column),
            "the use site and the declaration must be distinguishable"
        );
        let (line, column) = position_of(&code, "MARKER_ALPHA");
        let token = map
            .lookup_token(&table, line, column.saturating_sub(1))
            .expect("the sampled minified position must be mapped");
        assert_eq!(
            (
                token.get_source_id().and_then(|id| map.get_source(id)),
                token.get_src_line(),
                token.get_src_col(),
            ),
            (Some("diffpack:///a.ts"), inlined_line, inlined_column),
            "the minified literal must resolve to the `marker` use at a.ts {}:{inlined_column}, got {:?}",
            inlined_line + 1,
            (token.get_src_line(), token.get_src_col()),
        );
        let _ = (marker_line, marker_column);

        // The MANGLED function binding resolves to the original declaration AND
        // recovers its original NAME — the whole point of `names` in a production
        // map, and something no line-granular map can provide.
        let mangled = code
            .split_once("function ")
            .map(|(_, rest)| {
                rest.chars()
                    .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
                    .collect::<String>()
            })
            .expect("the minified chunk declares the hoisted function");
        assert_ne!(mangled, "greet", "the minifier must have renamed it");
        let (line, column) = position_of(&code, &format!("function {mangled}"));
        let token = map
            .lookup_token(&table, line, column + "function ".len() as u32)
            .expect("the mangled binding must be mapped");
        assert_eq!(
            (token.get_src_line(), token.get_src_col()),
            (greet_line, greet_column),
            "the mangled binding must resolve to a.ts {}:{greet_column}",
            greet_line + 1,
        );
        assert_eq!(
            token.get_name_id().and_then(|id| map.get_name(id)),
            Some("greet"),
            "the composed map must recover the original identifier for a mangled name"
        );
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
        let on_disk = EmitSummary::of(&public_dir).unwrap();
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

    /// The client `public/` build must emit BROWSER-executable ESM: the entry
    /// `client.js` is injected by the SSR document as
    /// `<script type="module" src="/client.js">`, so a CommonJS `module.exports=…`
    /// entry throws `module is not defined` under the ESM goal and the app never
    /// hydrates. This builds a small app with a Node built-in external (forcing
    /// the shared registry runtime and thus the browser `requireNative` stub) and
    /// a dynamic import (a split chunk), emits it via `emit_public`, then LOADS
    /// the entry with `import()` under `node` (as an ESM oracle) and asserts the
    /// entry's top-level code ran — proving there is no `module is not defined`
    /// and no `node:module` import a browser could not resolve.
    #[test]
    fn emit_public_entry_loads_as_a_browser_es_module_under_node() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("lazy.js"),
            "export const lazy = 'lazy-value';\n",
        )
        .unwrap();
        // `import os from 'node:os'` forces the runtime path (the flat path cannot
        // bind an external); it is used only inside a function, so module init
        // never calls the browser stub. The dynamic import forces a split chunk.
        fs::write(
            directory.path().join("entry.js"),
            "import os from 'node:os';\n\
             export function platform(){ return os.platform(); }\n\
             globalThis.__diffpack_client_ran = true;\n\
             import('./lazy.js').then((m) => { globalThis.__diffpack_lazy = m.lazy; });\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output_root = directory.path().join(".diffpack-output");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler
            .emit_public(&reachable, &output_root, EmitOptions::default())
            .unwrap();

        let public_dir = output_root.join("public");
        let client = public_dir.join("client.js");
        // Every emitted client `.js` passes `node --check` under the ESM goal.
        for entry in fs::read_dir(&public_dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|value| value.to_str()) == Some("js") {
                node_check(&path);
            }
        }
        // The browser entry has NO `node:module` import and DOES `export default`.
        let code = fs::read_to_string(&client).unwrap();
        assert!(
            !code.contains("node:module"),
            "browser ESM entry must not import node:module"
        );
        assert!(
            code.contains("export default"),
            "browser ESM entry must export a default"
        );

        // Load the entry as a real ES module. A CJS entry would throw
        // `module is not defined`; a `node:module` import would fail to resolve.
        let harness = public_dir.join("harness.mjs");
        fs::write(
            &harness,
            // The `setTimeout` lets the entry's `import('./lazy.js')` settle before
            // the split chunk's value is asserted. Loading is not enough: a flat
            // chunk consumed through the registry protocol resolves to `undefined`
            // rather than throwing, so a load-only assertion passes while the
            // dynamic import silently yields nothing.
            "import(process.argv[2]).then(() => new Promise((done) => setTimeout(done, 0))).then(() => { if (globalThis.__diffpack_client_ran !== true) { console.error('entry top-level did not run'); process.exit(3); } if (globalThis.__diffpack_lazy !== 'lazy-value') { console.error('SPLIT_CHUNK_VALUE:' + String(globalThis.__diffpack_lazy)); process.exit(5); } console.log('LOADED'); }).catch((e) => { console.error('LOAD_ERROR:' + e.message); process.exit(4); });\n",
        )
        .unwrap();
        let output = node_command()
            .arg(&harness)
            .arg(&client)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success() && stdout.contains("LOADED"),
            "client.js did not load as an ES module: stdout={stdout} stderr={stderr}"
        );
        assert!(
            !stderr.contains("module is not defined"),
            "`module is not defined` leaked: {stderr}"
        );
    }

    fn run_node(path: &Path) -> String {
        let output = node_command().arg(path).output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap()
    }

    /// Syntax-checks a file as JavaScript under the Node ESM goal. `node --check`
    /// is a build oracle only, never in the build path.
    fn node_check(path: &Path) {
        let output = node_command()
            .arg("--check")
            .arg(path)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "node --check failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn emit_server_writes_an_mjs_layout_that_node_accepts() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("style.css"), ".a { color: red; }").unwrap();
        fs::write(
            directory.path().join("lazy.js"),
            "export const lazy = 'lazy';",
        )
        .unwrap();
        fs::write(
            directory.path().join("server.ts"),
            "import './style.css';\n\
             console.log('render');\n\
             import('./lazy.js').then(({ lazy }) => console.log(lazy));\n",
        )
        .unwrap();

        let entry = directory.path().join("server.ts");
        let output_root = directory.path().join(".diffpack-output");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let summary = bundler
            .emit_server(&reachable, &output_root, EmitOptions::default())
            .unwrap();

        // The server entry plus its dynamically imported chunk, as `.mjs`.
        assert!(
            summary.javascript_files >= 2,
            "expected the server entry and a dynamic chunk: {summary:?}"
        );

        let server_dir = output_root.join("server");
        assert!(server_dir.join("server.mjs").is_file());
        assert!(
            server_dir.join("server.chunk-1.mjs").is_file(),
            "the dynamic import lands in an `.mjs` chunk"
        );
        // No stray `.js` in the server build: everything is Node ESM.
        assert_eq!(summary.output_dir, server_dir);

        // Every emitted `.mjs` must be syntactically valid under Node's ESM goal.
        for entry in fs::read_dir(&server_dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|value| value.to_str()) == Some("mjs") {
                node_check(&path);
            }
        }

        // A re-emit rebuilds `server/` from scratch.
        let stale = server_dir.join("stale.mjs");
        fs::write(&stale, "// stale").unwrap();
        bundler
            .emit_server(&reachable, &output_root, EmitOptions::default())
            .unwrap();
        assert!(!stale.exists(), "re-emit must clear stale output");
    }

    /// The server `.mjs` output must not merely pass `node --check`; it must
    /// EXECUTE under Node's ESM goal. This builds a small multi-module app with a
    /// static cross-module import, an external Node built-in (forcing the shared
    /// Top-level `await` cannot exist in CommonJS output or inside the factory
    /// runtime; both must be hard, module-naming errors (previously the build
    /// "succeeded" and emitted a bundle Node rejects at parse — the conformance
    /// suite's worst honesty finding). In single-chunk ESM output it is
    /// representable and must actually run.
    #[test]
    fn top_level_await_is_a_hard_error_in_cjs_and_runs_in_flat_esm() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("value.js"),
            "export const value = await Promise.resolve('tla-value');\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { value } from './value.js';\nconsole.log('got:' + value);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();

        let error = bundler
            .emit_with_options(
                &reachable,
                &directory.path().join("dist/out.js"),
                EmitOptions::default(),
            )
            .unwrap_err();
        assert!(error.contains("top-level await"), "{error}");
        assert!(error.contains("value.js"), "names the module: {error}");
        assert!(error.contains("--format esm"), "names the way out: {error}");

        let esm_out = directory.path().join("dist/out.mjs");
        bundler
            .emit_with_options(
                &reachable,
                &esm_out,
                EmitOptions {
                    format: ModuleFormat::Esm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();
        assert_eq!(run_node(&esm_out), "got:tla-value\n");
    }

    /// Top-level `await` in a CODE-SPLIT ESM build. Splitting forces the registry
    /// runtime, whose factories used to be plain synchronous functions — so this
    /// whole graph was a hard "requires the single-chunk scope-hoisted ESM output"
    /// error, which is what cal.com's SSR bundle hit through
    /// `i18next-fs-backend` (`await import('node:fs')` at module scope).
    ///
    /// The awaiting module now renders as an `async` factory, and the property
    /// propagates up the static import edges: `value.js` awaits, so `middle.js`
    /// (which imports it) and the entry (which imports that) are async too, and
    /// each of their import sites awaits. The bundle must EXECUTE under Node and
    /// print the awaited value — not merely parse — and the dynamically imported
    /// chunk (which is what forces the split) must resolve to the finished
    /// namespace of its own async module.
    #[test]
    fn top_level_await_runs_in_a_code_split_esm_build() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("value.js"),
            "export const value = await Promise.resolve('tla-value');\n",
        )
        .unwrap();
        // A second hop, so the async property has to PROPAGATE rather than only
        // apply to the module that literally awaits.
        fs::write(
            directory.path().join("middle.js"),
            "import { value } from './value.js';\nexport const shouted = value.toUpperCase();\n",
        )
        .unwrap();
        // The dynamically imported module also awaits: `require.dynamic` must
        // resolve through the async path, or `lazy` reads back `undefined`.
        fs::write(
            directory.path().join("lazy.js"),
            "export const lazy = await Promise.resolve('lazy-value');\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { shouted } from './middle.js';\n\
             const { lazy } = await import('./lazy.js');\n\
             console.log('got:' + shouted + ':' + lazy);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();

        let esm_out = directory.path().join("dist/out.mjs");
        let stats = bundler
            .emit_with_options(
                &reachable,
                &esm_out,
                EmitOptions {
                    format: ModuleFormat::Esm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();
        assert!(
            stats.written.len() >= 2,
            "the dynamic import must split off a chunk: {:?}",
            stats.written
        );
        for path in &stats.written {
            node_check(path);
        }
        let code = fs::read_to_string(&esm_out).unwrap();
        assert!(
            code.contains("async function(module,exports,require"),
            "an awaiting module renders as an async factory: {code}"
        );
        assert!(
            code.contains("await require.esmAsync(\"./value.js\")"),
            "the importer awaits its async dependency: {code}"
        );
        assert_eq!(run_node(&esm_out), "got:TLA-VALUE:lazy-value\n");
    }

    /// The same graph under the DEV (HMR) runtime, which used to reject it outright
    /// ("top-level await ... is not supported in a dev (HMR) build"). That refusal is
    /// what stopped `diffpack dev` on cal.com: its SSR graph reaches
    /// `i18next-fs-backend`, whose `readFile.js` does `await import('node:fs')` at
    /// module scope, so the whole dev server died before serving a request.
    ///
    /// The async machinery and the HMR machinery are independent and must compose:
    /// the HMR runtime has to publish `requireAsync` (a chunk whose root is async
    /// returns `__runtime.requireAsync(...)`), and its version-aware
    /// `require.dynamic` has to resolve through it. Node EXECUTING the bundle —
    /// including the dynamically imported, separately-chunked async module — is the
    /// assertion.
    #[test]
    fn top_level_await_runs_under_the_dev_hmr_runtime() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("value.js"),
            "export const value = await Promise.resolve('tla-value');\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("middle.js"),
            "import { value } from './value.js';\nexport const shouted = value.toUpperCase();\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("lazy.js"),
            "export const lazy = await Promise.resolve('lazy-value');\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { shouted } from './middle.js';\n\
             const { lazy } = await import('./lazy.js');\n\
             console.log('got:' + shouted + ':' + lazy);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();

        let esm_out = directory.path().join("dist/out.mjs");
        let stats = bundler
            .emit_with_options(
                &reachable,
                &esm_out,
                EmitOptions {
                    format: ModuleFormat::Esm,
                    hmr: true,
                    ..EmitOptions::default()
                },
            )
            .unwrap();
        assert!(
            stats.written.len() >= 2,
            "the dynamic import must split off a chunk: {:?}",
            stats.written
        );
        for path in &stats.written {
            node_check(path);
        }
        let code = fs::read_to_string(&esm_out).unwrap();
        assert!(
            code.contains("requireAsync:__requireAsync,"),
            "the HMR runtime must publish requireAsync for an async chunk root: {code}"
        );
        assert!(
            code.contains("__requireAsync(chunk[1])"),
            "the version-aware dynamic require must resolve through the async path: {code}"
        );
        assert_eq!(run_node(&esm_out), "got:TLA-VALUE:lazy-value\n");
    }

    /// A hot update whose re-run reaches an ASYNC module must not report success (or
    /// publish a fresh SSR handler) until that module's top-level `await` has SETTLED.
    ///
    /// This is the exact hazard the old blanket refusal cited. `__require` returns a
    /// module's exports object synchronously in both cases — the object exists before
    /// the factory's first `await` — so a naive re-run looks like it worked while the
    /// module body is still suspended, and the dev server hands the next SSR request a
    /// half-initialised entry.
    ///
    /// The awaited work here is a TIMER, not a resolved promise, so no amount of
    /// microtask draining can accidentally make the assertion pass: only a real `await`
    /// of the module's pending initialisation does.
    #[test]
    fn a_hot_update_waits_for_an_async_modules_top_level_await() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let value_js = directory.path().join("value.js");
        fs::write(
            &value_js,
            "export const value = await new Promise(r => setTimeout(() => r('v1'), 20));\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { value } from './value.js';\n\
             (globalThis.__log ??= []).push(value);\n\
             export const label = value;\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let options = EmitOptions {
            format: ModuleFormat::Esm,
            hmr: true,
            ..EmitOptions::default()
        };
        let esm_out = directory.path().join("dist/out.mjs");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler
            .emit_with_options(&reachable, &esm_out, options)
            .unwrap();

        // The edit, exactly as the dev server applies one: re-discover, locate the
        // changed module's runtime id, and render the tiny register-only HMR chunk
        // carrying only its new factory.
        fs::write(
            &value_js,
            "export const value = await new Promise(r => setTimeout(() => r('v2'), 20));\n",
        )
        .unwrap();
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let changed = BTreeSet::from([ModuleId::from(
            value_js.canonicalize().unwrap().to_string_lossy().as_ref(),
        )]);
        let located = bundler
            .hmr_locate(&reachable, &changed, "out.mjs")
            .unwrap();
        assert_eq!(located.len(), 1, "the edited module must be located");
        let runtime_id = located[0].runtime_id;
        let hmr_path = directory.path().join("dist/hmr-1.mjs");
        assert!(
            bundler
                .write_hmr_chunk(&reachable, &changed, "out.mjs", options, ModuleFormat::Esm, &hmr_path)
                .unwrap(),
            "the edited module is live, so it renders"
        );
        node_check(&hmr_path);

        // Drive the update the way the Node control endpoint does: register the new
        // factory, then `serverInvalidate`, which re-runs the entry in-process and
        // republishes the SSR handler.
        let harness = directory.path().join("dist/harness.mjs");
        fs::write(
            &harness,
            format!(
                "import './out.mjs';\n\
                 await import('./hmr-1.mjs?__diffpack_hmr=1');\n\
                 const rt = globalThis.__diffpack_hmr_runtime;\n\
                 await rt.serverInvalidate([{runtime_id}], []);\n\
                 console.log(JSON.stringify({{\n\
                 log: globalThis.__log,\n\
                 published: globalThis.__diffpack_ssr_entry.label,\n\
                 }}));\n"
            ),
        )
        .unwrap();
        assert_eq!(
            run_node(&harness),
            "{\"log\":[\"v1\",\"v2\"],\"published\":\"v2\"}\n",
            "the hot update must observe the re-run module's SETTLED top-level await"
        );
    }

    /// Every dev-client module's Fast Refresh registrations must be NAMESPACED by
    /// that module, so two modules that happen to define a same-named component are
    /// never mistaken for two versions of one component.
    ///
    /// oxc's refresh transform emits `$RefreshReg$(_c, "Widget")` — the local name
    /// only — and react-refresh keys families in ONE global map, so an unscoped id
    /// makes the second registration read as a hot update of the first. On cal.com
    /// that put hundreds of phantom updates in the queue before a single edit; the
    /// first real edit then swapped unrelated component types into the live tree and
    /// React's `scheduleRefresh` -> `flushSyncWork` loop never terminated, wedging the
    /// browser tab. This bundles two same-named components, RUNS the emitted dev
    /// bundle against a recording refresh runtime, and asserts the ids it registers
    /// are distinct and carry their own module's path.
    #[test]
    fn a_dev_client_bundle_scopes_every_fast_refresh_registration_to_its_module() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let root = directory.path();
        write_jsx_runtime_package(root, "react");
        fs::write(
            root.join("list.jsx"),
            "export function Widget() { return <div>list</div>; }\n",
        )
        .unwrap();
        fs::write(
            root.join("table.jsx"),
            "export function Widget() { return <div>table</div>; }\n",
        )
        .unwrap();
        fs::write(
            root.join("entry.jsx"),
            "import { Widget as FromList } from './list.jsx';\n\
             import { Widget as FromTable } from './table.jsx';\n\
             (globalThis.__used ??= []).push(FromList, FromTable);\n",
        )
        .unwrap();
        let entry = root.join("entry.jsx");

        // The dev build: the bundler's own `hmr` flag is what turns the per-module
        // refresh instrumentation on (`build-app` never sets it).
        let dev_config = BuildConfig {
            hmr: true,
            target: Target::Client,
            ..BuildConfig::default()
        };
        let (bundler, update) = Bundler::discover_direct_with_config(&entry, &dev_config).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let dev_out = root.join("dev/out.mjs");
        bundler
            .emit_with_options(
                &reachable,
                &dev_out,
                EmitOptions {
                    format: ModuleFormat::Esm,
                    hmr: true,
                    ..EmitOptions::default()
                },
            )
            .unwrap();

        let dev_code = fs::read_to_string(&dev_out).unwrap();
        assert!(
            dev_code.contains("$RefreshReg$"),
            "the dev build must instrument its components: {dev_code}"
        );
        let harness = root.join("dev/harness.mjs");
        fs::write(
            &harness,
            "globalThis.window = globalThis;\n\
             const ids = [];\n\
             globalThis.$RefreshRuntime$ = {\n\
             register: (type, id) => ids.push(id),\n\
             createSignatureFunctionForTransform: () => (type) => type,\n\
             registerExportsForReactRefresh: () => {},\n\
             validateRefreshBoundaryAndEnqueueUpdate: () => undefined,\n\
             };\n\
             await import('./out.mjs');\n\
             console.log(JSON.stringify(ids));\n",
        )
        .unwrap();
        let registered: Vec<String> = serde_json::from_str(run_node(&harness).trim()).unwrap();
        assert_eq!(
            registered.len(),
            2,
            "both modules must register their component: {registered:?}"
        );
        assert_ne!(
            registered[0], registered[1],
            "same-named components in different modules must not share a family: {registered:?}"
        );
        for (module, id) in [("list.jsx", &registered[0]), ("table.jsx", &registered[1])] {
            assert!(
                id.contains(module) && id.ends_with(" Widget"),
                "the family id must be its own module plus the export name: {id}"
            );
        }

        // Production is untouched: no refresh instrumentation is emitted at all, so
        // none of this can reach a `build-app` bundle.
        let (production, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let production_out = root.join("dist/out.mjs");
        production
            .emit_with_options(
                &production.reachable_modules_direct(),
                &production_out,
                EmitOptions {
                    format: ModuleFormat::Esm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();
        let code = fs::read_to_string(&production_out).unwrap();
        assert!(
            !code.contains("$RefreshReg$") && !code.contains("$RefreshRuntime$"),
            "a production bundle must carry no Fast Refresh instrumentation: {code}"
        );
    }

    /// An imported binding must be initialized before ANY of the module's body runs,
    /// even a statement written ABOVE the import.
    ///
    /// `import` declarations are hoisted by the language: the spec instantiates and
    /// evaluates every requested module before the importer's body executes, so source
    /// position says nothing about when a binding becomes available. Babel's JSX-pragma
    /// output relies on it — `var __jsx = React.createElement;` is emitted above
    /// `import React from "react"` (next-i18next's `appWithTranslation.js` ships exactly
    /// that) — and lowering each import in place made the binding read `undefined`,
    /// failing with `TypeError: Cannot convert undefined or null to object` inside a
    /// render, on code that is perfectly valid ESM.
    #[test]
    fn an_import_binding_is_initialized_before_a_statement_written_above_it() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("dep.js"),
            "export default { make: () => 'made' };\nexport const named = 'named';\n",
        )
        .unwrap();
        // Babel's JSX-pragma shape verbatim: a body statement above the import.
        fs::write(
            directory.path().join("entry.js"),
            "const make = Dep.make;\n\
             import Dep, { named } from './dep.js';\n\
             console.log(make() + ':' + named);\n",
        )
        .unwrap();
        assert_eq!(bundle_and_run(directory.path()), "made:named\n");
    }

    /// The same rule for a bare side-effect `import`: the requested module runs before
    /// the importer's body, not at the import statement's source position.
    #[test]
    fn a_side_effect_import_runs_before_the_body_that_precedes_it() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("effect.js"),
            "globalThis.__order = (globalThis.__order || '') + 'effect';\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "globalThis.__order = (globalThis.__order || '') + 'body';\n\
             import './effect.js';\n\
             console.log(globalThis.__order);\n",
        )
        .unwrap();
        assert_eq!(bundle_and_run(directory.path()), "effectbody\n");
    }

    /// A specifier one module reaches BOTH statically and dynamically is not a
    /// code-split boundary, and moving it into a lazily-fetched chunk breaks the
    /// static reference.
    ///
    /// The shape is the ordinary lazy-component barrel:
    ///
    /// ```js
    /// export { default as Foo } from "./Foo";
    /// export const FooLazy = dynamic(() => import("./Foo"));
    /// ```
    ///
    /// Reading only the `import()` said "./Foo is a chunk root", so the module moved
    /// out of the entry's static closure — and then the `export … from` on the line
    /// above, which lowers to a synchronous registry lookup, threw
    /// `Module is not loaded: <id>` the first time the barrel evaluated.
    ///
    /// Node EXECUTING the bundle is the assertion; the chunk-count check pins that the
    /// build really is code-split (so the test cannot pass by not splitting at all).
    #[test]
    fn a_specifier_reached_both_statically_and_dynamically_is_not_split_off() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("shared.js"),
            "export const label = 'shared-label';\nexport default 'shared-default';\n",
        )
        .unwrap();
        // An unrelated module reached ONLY dynamically, so the build genuinely splits.
        fs::write(
            directory.path().join("only-lazy.js"),
            "export const only = 'only-lazy';\n",
        )
        .unwrap();
        // The barrel: a static re-export AND a dynamic import of the SAME specifier.
        fs::write(
            directory.path().join("barrel.js"),
            "export { label } from './shared.js';\n\
             export const lazyShared = () => import('./shared.js');\n\
             export const lazyOther = () => import('./only-lazy.js');\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { label, lazyShared, lazyOther } from './barrel.js';\n\
             Promise.all([lazyShared(), lazyOther()]).then(([a, b]) =>\n\
             console.log(label + ':' + a.label + ':' + b.only));\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();

        let esm_out = directory.path().join("dist/out.mjs");
        let stats = bundler
            .emit_with_options(
                &reachable,
                &esm_out,
                EmitOptions {
                    format: ModuleFormat::Esm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();
        assert!(
            stats.written.len() >= 2,
            "the dynamic-only import must still split off a chunk: {:?}",
            stats.written
        );
        for path in &stats.written {
            node_check(path);
        }
        // The both-ways module belongs to the entry chunk; only the dynamic-ONLY one
        // may live in a split chunk.
        for path in &stats.written {
            if path == &esm_out {
                continue;
            }
            let chunk = fs::read_to_string(path).unwrap();
            assert!(
                !chunk.contains("shared-label"),
                "a statically-referenced module must not be moved into a lazy chunk: {}",
                path.display()
            );
        }
        assert_eq!(
            run_node(&esm_out),
            "shared-label:shared-label:only-lazy\n"
        );
    }

    /// The same hole through a `require(...)`: a synchronous read of a module that is
    /// also `import()`ed elsewhere. `require` returns the exports immediately, so the
    /// target can never sit behind a chunk fetch.
    #[test]
    fn a_specifier_reached_by_require_and_by_dynamic_import_is_not_split_off() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("shared.js"),
            "exports.label = 'req-shared';\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("only-lazy.js"),
            "export const only = 'only-lazy';\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "export const eager = require('./shared.js').label;\n\
             export const lazy = () => import('./shared.js');\n\
             Promise.all([lazy(), import('./only-lazy.js')]).then(([a, b]) =>\n\
             console.log(eager + ':' + b.only));\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let esm_out = directory.path().join("dist/out.mjs");
        let stats = bundler
            .emit_with_options(
                &reachable,
                &esm_out,
                EmitOptions {
                    format: ModuleFormat::Esm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();
        for path in &stats.written {
            node_check(path);
        }
        assert_eq!(run_node(&esm_out), "req-shared:only-lazy\n");
    }

    /// `export * from "./x"` where `./x` is tree-shaken away must not leave a
    /// runtime lookup for `./x` behind.
    ///
    /// The registry's miss path is how EXTERNALS work (`node:fs`, an uninstalled
    /// optional dependency), so a lookup for a module the bundle dropped does not
    /// fail the build — it becomes a raw `require("./x")` in the emitted file and
    /// throws MODULE_NOT_FOUND the moment the module is evaluated. `tslog` ships
    /// exactly this shape: an `interfaces.js` whose entire body is `export {}`,
    /// star-re-exported by its logger. Node EXECUTING the bundle is the assertion
    /// that matters; the byte check pins the cause.
    #[test]
    fn a_star_reexport_of_a_shaken_away_module_leaves_no_runtime_lookup() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        // `sideEffects: false` is what entitles dead-module elimination to drop a
        // module nothing demands an export of — the declaration every package that
        // ships this shape (tslog among them) makes.
        fs::write(
            directory.path().join("package.json"),
            r#"{"name":"star-reexport-fixture","sideEffects":false}"#,
        )
        .unwrap();
        // Type-only in spirit and empty in fact: nothing to export, no side
        // effects, so dead-module elimination is entitled to drop it entirely.
        fs::write(directory.path().join("interfaces.js"), "export {};\n").unwrap();
        fs::write(
            directory.path().join("logger.js"),
            "export * from './interfaces.js';\nexport const log = (message) => 'log:' + message;\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { log } from './logger.js';\nconsole.log(log('ok'));\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let out = directory.path().join("dist/out.mjs");
        bundler
            .emit_with_options(
                &reachable,
                &out,
                EmitOptions {
                    format: ModuleFormat::Esm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();
        let code = fs::read_to_string(&out).unwrap();
        assert!(
            !code.contains("./interfaces.js"),
            "the dropped module must not be referenced by the emitted code: {code}"
        );
        assert_eq!(run_node(&out), "log:ok\n");
    }

    /// A bundle with no top-level `await` anywhere must be BYTE-IDENTICAL to what
    /// it was before async-module support: every async runtime line is gated on
    /// the build actually having one. Guards against the registry runtime growing
    /// dead weight (and against the async paths quietly turning on).
    #[test]
    fn a_build_without_top_level_await_emits_no_async_runtime() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("lazy.js"),
            "export const lazy = 'lazy';\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import('./lazy.js').then(({ lazy }) => console.log(lazy));\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, _) = Bundler::discover_direct(&entry).unwrap();
        let reachable = bundler.reachable_modules_direct();
        let esm_out = directory.path().join("dist/out.mjs");
        let stats = bundler
            .emit_with_options(
                &reachable,
                &esm_out,
                EmitOptions {
                    format: ModuleFormat::Esm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();
        for path in &stats.written {
            let code = fs::read_to_string(path).unwrap();
            for marker in [
                "__pending",
                "__requireAsync",
                "require.esmAsync",
                "require.async",
                "async function(module,exports",
                "await (async()=>{",
            ] {
                assert!(
                    !code.contains(marker),
                    "{}: a build with no top-level await must not emit {marker}",
                    path.display()
                );
            }
        }
    }

    /// A CommonJS `require()` cannot wait for a module that top-level-`await`s
    /// (Node itself throws `ERR_REQUIRE_ASYNC_MODULE`), and neither can the lazy
    /// getter `export * as ns from` lowers to. Both must be hard errors naming
    /// BOTH modules, never a bundle that reads a half-initialised namespace.
    #[test]
    fn reaching_an_async_module_without_an_awaitable_import_is_a_hard_error() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("value.js"),
            "export const value = await Promise.resolve('tla-value');\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("lazy.js"),
            "export const lazy = 'lazy';\n",
        )
        .unwrap();

        // (1) A CommonJS `require` of the awaiting module.
        fs::write(
            directory.path().join("entry.js"),
            "const { value } = require('./value.js');\n\
             console.log(value);\n\
             import('./lazy.js');\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, _) = Bundler::discover_direct(&entry).unwrap();
        let reachable = bundler.reachable_modules_direct();
        let error = bundler
            .emit_with_options(
                &reachable,
                &directory.path().join("dist/out.mjs"),
                EmitOptions {
                    format: ModuleFormat::Esm,
                    ..EmitOptions::default()
                },
            )
            .unwrap_err();
        assert!(error.contains("top-level await"), "{error}");
        assert!(error.contains("value.js"), "names the async module: {error}");
        assert!(error.contains("entry.js"), "names the importer: {error}");
        assert!(
            error.contains("ERR_REQUIRE_ASYNC_MODULE"),
            "names Node's own diagnosis: {error}"
        );

        // (2) `export * as ns from` the awaiting module — a lazy getter.
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("value.js"),
            "export const value = await Promise.resolve('tla-value');\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("lazy.js"),
            "export const lazy = 'lazy';\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "export * as values from './value.js';\nimport('./lazy.js');\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, _) = Bundler::discover_direct(&entry).unwrap();
        let reachable = bundler.reachable_modules_direct();
        let error = bundler
            .emit_with_options(
                &reachable,
                &directory.path().join("dist/out.mjs"),
                EmitOptions {
                    format: ModuleFormat::Esm,
                    ..EmitOptions::default()
                },
            )
            .unwrap_err();
        assert!(error.contains("export * as"), "{error}");
        assert!(error.contains("value.js"), "names the async module: {error}");
        assert!(error.contains("entry.js"), "names the importer: {error}");
    }

    /// `import.meta` is a syntax error anywhere in a CommonJS file, so CJS
    /// output must refuse; in ESM output it stays, resolving against the
    /// emitted chunk (the standard bundler semantic).
    #[test]
    fn import_meta_is_a_hard_error_in_cjs_and_survives_in_esm() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "console.log('url-kind:' + (import.meta.url.startsWith('file://')));\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();

        let error = bundler
            .emit_with_options(
                &reachable,
                &directory.path().join("dist/out.js"),
                EmitOptions::default(),
            )
            .unwrap_err();
        assert!(error.contains("import.meta"), "{error}");
        assert!(error.contains("entry.js"), "names the module: {error}");

        let esm_out = directory.path().join("dist/out.mjs");
        bundler
            .emit_with_options(
                &reachable,
                &esm_out,
                EmitOptions {
                    format: ModuleFormat::Esm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();
        assert_eq!(run_node(&esm_out), "url-kind:true\n");
    }

    /// Statement-level shaking must be TRANSITIVE: a pure helper (exported or
    /// not) referenced only by a dead export falls with it, through chains,
    /// while impure statements and everything they reference stay. Pinned by
    /// the realistic-corpus finding where non-exported helpers of dead exports
    /// made output 2.2x larger than esbuild's.
    #[test]
    fn shaking_drops_helpers_of_dead_exports_transitively() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("lib.js"),
            "const DEEP_CONFIG = { step: 3 };\n\
             function deepHelper(value) { return value + DEEP_CONFIG.step; }\n\
             function midHelper(value) { return deepHelper(value) * 2; }\n\
             export function unusedTool(value) { return midHelper(value); }\n\
             const KEPT_BASE = 40;\n\
             export function usedTool(value) { return value + KEPT_BASE; }\n\
             console.log('lib-side-effect:' + usedTool(0));\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { usedTool } from './lib.js';\nconsole.log('result:' + usedTool(2));\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler
            .emit_with_options(&reachable, &output, EmitOptions::default())
            .unwrap();
        let code = fs::read_to_string(&output).unwrap();
        for dead in ["unusedTool", "midHelper", "deepHelper", "DEEP_CONFIG"] {
            assert!(!code.contains(dead), "`{dead}` should be shaken:\n{code}");
        }
        for live in ["usedTool", "KEPT_BASE", "lib-side-effect"] {
            assert!(code.contains(live), "`{live}` must survive:\n{code}");
        }
        assert_eq!(run_node(&output), "lib-side-effect:40\nresult:42\n");
    }

    /// Vite's `assetsInlineLimit`: in Vite mode a small asset import yields a
    /// `data:` URI (no emitted file, no request); over the limit — or with the
    /// limit disabled (generic bundling) — it stays a hashed public file.
    #[test]
    fn small_assets_inline_as_data_uris_only_in_vite_mode() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("icon.svg"), "<svg xmlns='x'/>").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import icon from './icon.svg';\nconsole.log(icon.slice(0, 30));\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");

        let inline_config = BuildConfig {
            asset_inline_limit: 4096,
            ..BuildConfig::default()
        };
        let (bundler, update) = Bundler::discover_direct_with_config(&entry, &inline_config).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = directory.path().join("dist/bundle.js");
        bundler
            .emit_with_options(&reachable, &output, EmitOptions::default())
            .unwrap();
        assert_eq!(run_node(&output), "data:image/svg+xml,%3csvg%20xm\n");
        assert!(
            !directory.path().join("dist/assets").exists(),
            "an inlined asset emits no file"
        );

        let (bundler, _) = Bundler::discover_direct(&entry).unwrap();
        let reachable = bundler.reachable_modules_direct();
        let plain = directory.path().join("dist-plain/bundle.js");
        bundler
            .emit_with_options(&reachable, &plain, EmitOptions::default())
            .unwrap();
        assert!(
            run_node(&plain).starts_with("/assets/icon-"),
            "generic bundling keeps the hashed file URL"
        );
    }

    /// `new Worker(new URL('./x', import.meta.url))` bundles the worker entry
    /// as its own self-contained file under `assets/` and substitutes its
    /// public URL — shipping the raw specifier would 404 at runtime (found
    /// live on wall-go's minimax AI workers).
    #[test]
    fn module_workers_are_bundled_and_their_urls_substituted() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("logic.js"),
            "export function answer() { return 42; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("worker.js"),
            "import { answer } from './logic.js';\nself.postMessage(answer());\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "const w = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });\nconsole.log('spawned:' + (w instanceof Object));\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = directory.path().join("dist/bundle.js");
        bundler
            .emit_with_options(
                &reachable,
                &output,
                EmitOptions {
                    format: ModuleFormat::BrowserEsm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();
        let code = fs::read_to_string(&output).unwrap();
        assert!(!code.contains("__diffpack_worker__"), "placeholder replaced: {code}");
        assert!(!code.contains("./worker.js"), "raw specifier gone: {code}");
        let url_start = code.find("/assets/worker-").expect("worker URL substituted");
        let url = code[url_start..].split(['"', '\'', '`']).next().unwrap();
        let emitted = directory.path().join("dist").join(url.trim_start_matches('/'));
        assert!(emitted.is_file(), "worker bundle emitted at {}", emitted.display());
        let worker_code = fs::read_to_string(&emitted).unwrap();
        assert!(worker_code.contains("postMessage"), "{worker_code}");
        assert!(worker_code.contains("42"), "the worker's import is bundled in: {worker_code}");
    }

    /// Side-effect imports must execute in IMPORT order, not module-id order.
    /// The entry imports `./bbb.js` before `./aaa.js`; alphabetical ordering
    /// would run `aaa` first, which is exactly the bug this pins down (the
    /// conformance suite's `order-side-effect-imports` finding).
    #[test]
    fn side_effect_imports_execute_in_import_order_not_id_order() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("aaa.js"), "console.log('aaa');\n").unwrap();
        fs::write(directory.path().join("bbb.js"), "console.log('bbb');\n").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './bbb.js';\nimport './aaa.js';\nconsole.log('entry');\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler
            .emit_with_options(&reachable, &output, EmitOptions::default())
            .unwrap();
        assert_eq!(run_node(&output), "bbb\naaa\nentry\n");
    }

    /// The emitted stylesheet must follow the same execution order, because the
    /// CSS cascade breaks equal-specificity ties by document order: a rule from
    /// a stylesheet the entry imports FIRST must lose to a same-specificity rule
    /// imported later, no matter how the module paths sort. (Found live on the
    /// create-vite fixture: `App.css`'s `.counter` override lost to `index.css`
    /// because alphabetical order inverted the cascade.)
    #[test]
    fn extracted_css_follows_import_order_so_the_cascade_ties_break_correctly() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("zzz-base.css"), ".x { color: red; }\n").unwrap();
        fs::write(directory.path().join("aaa-widget.css"), ".x { color: blue; }\n").unwrap();
        fs::write(
            directory.path().join("aaa-widget.js"),
            "import './aaa-widget.css';\nexport const widget = 1;\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './zzz-base.css';\nimport { widget } from './aaa-widget.js';\nconsole.log(widget);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler
            .emit_with_options(&reachable, &output, EmitOptions::default())
            .unwrap();
        let stylesheet = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        let base_at = stylesheet.find("red").expect("base rule present");
        let widget_at = stylesheet.find("blue").expect("widget rule present");
        assert!(
            base_at < widget_at,
            "entry-imported stylesheet must precede the later component's:\n{stylesheet}"
        );
    }

    /// registry runtime), and a dynamic `import()` of a split chunk, emits it via
    /// the server path, then runs the entry under `node` and asserts both the
    /// static value and the dynamically-loaded chunk's value reach stdout.
    #[test]
    fn emit_server_mjs_executes_the_entry_and_dynamic_chunk_under_node() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("util.js"),
            "export const base = 10;\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("lazy.js"),
            "import os from 'node:os';\n\
             export const lazy = 'lazy-value';\n\
             export function describe(){ return typeof os.platform === 'function' ? 'has-os' : 'no-os'; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("server.ts"),
            "import path from 'node:path';\n\
             import { base } from './util.js';\n\
             console.log('base:' + base);\n\
             console.log('sep:' + (path.sep.length === 1));\n\
             import('./lazy.js').then((m) => { console.log('lazy:' + m.lazy + ':' + m.describe()); });\n",
        )
        .unwrap();

        let entry = directory.path().join("server.ts");
        let output_root = directory.path().join(".diffpack-output");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler
            .emit_server(&reachable, &output_root, EmitOptions::default())
            .unwrap();

        let server_entry = output_root.join("server/server.mjs");
        assert!(
            output_root.join("server/server.chunk-1.mjs").is_file(),
            "the dynamic import lands in its own `.mjs` chunk"
        );
        // Actually run it: `module is not defined` would abort here, so a clean
        // stdout proves the emitted ESM genuinely executes.
        assert_eq!(
            run_node(&server_entry),
            "base:10\nsep:true\nlazy:lazy-value:has-os\n"
        );
    }

    /// A host that wants a FRESH module graph re-imports the entry under a new URL
    /// after dropping the runtime globals — the react-server `serve` worker's
    /// protocol. The registry lives on `globalThis`, so the new entry instance
    /// builds a new registry; every chunk it dynamically imports must therefore be
    /// a new instance too, or the chunk stays in Node's ESM cache, never re-runs
    /// its `__register`, and `__require` throws "Module is not loaded: <id>".
    #[test]
    fn a_fresh_entry_instance_gets_fresh_chunk_instances() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("lazy.js"),
            "export const lazy = 'lazy-value';\n",
        )
        .unwrap();
        // The dynamic import fires during the entry's own evaluation, which is what
        // gets the chunk into the ESM cache before the re-import happens.
        fs::write(
            directory.path().join("server.ts"),
            "export const loaded = import('./lazy.js').then((m) => m.lazy);\n\
             export async function read(){ return await loaded; }\n",
        )
        .unwrap();

        let entry = directory.path().join("server.ts");
        let output_root = directory.path().join(".diffpack-output");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler
            .emit_server(&reachable, &output_root, EmitOptions::default())
            .unwrap();
        assert!(
            output_root.join("server/server.chunk-1.mjs").is_file(),
            "the dynamic import must land in its own chunk for this test to mean anything"
        );

        let driver = directory.path().join("driver.mjs");
        fs::write(
            &driver,
            "import { pathToFileURL } from 'node:url';\n\
             const url = pathToFileURL(process.argv[2]).href;\n\
             const first = await import(url);\n\
             console.log('first:' + await (first.default || first).read());\n\
             for (const key of Object.keys(globalThis)) {\n\
               if (key.indexOf('__diffpack_runtime:') === 0) delete globalThis[key];\n\
             }\n\
             const second = await import(url + '?v=2');\n\
             console.log('second:' + await (second.default || second).read());\n",
        )
        .unwrap();
        let executed = node_command()
            .arg(&driver)
            .arg(output_root.join("server/server.mjs"))
            .output()
            .unwrap();
        assert!(
            executed.status.success(),
            "{}",
            String::from_utf8_lossy(&executed.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&executed.stdout),
            "first:lazy-value\nsecond:lazy-value\n"
        );
    }

    /// Polls `127.0.0.1:port` until it accepts a connection (or the attempts run
    /// out), then makes one `HTTP/1.0` GET and returns the full raw response.
    fn http_get_when_ready(port: u16, path: &str) -> String {
        use std::io::{Read, Write};
        use std::net::TcpStream;
        use std::time::Duration;
        let address = format!("127.0.0.1:{port}");
        for _ in 0..200 {
            if let Ok(mut stream) = TcpStream::connect(&address) {
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let request =
                    format!("GET {path} HTTP/1.0\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
                stream.write_all(request.as_bytes()).unwrap();
                let mut response = Vec::new();
                stream.read_to_end(&mut response).unwrap();
                return String::from_utf8_lossy(&response).into_owned();
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        panic!("server on port {port} never accepted a connection");
    }

    /// The emitted `server/index.mjs` must BOOT under Node and serve: SSR through
    /// the app's fetch handler (resolved from `server.mjs`'s CJS-interop default
    /// export by `_ssr/ssr.mjs`), plus a hashed asset from the sibling `public/`
    /// directory. Node is the runtime oracle — the request round-trips over real
    /// TCP, exactly like the acceptance runner.
    #[test]
    fn emitted_index_mjs_boots_and_serves_ssr_and_static_under_node() {
        use std::process::Stdio;
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let server_dir = directory.path().join("server");
        let public_dir = directory.path().join("public");
        fs::create_dir_all(&server_dir).unwrap();
        fs::create_dir_all(&public_dir).unwrap();

        // A stand-in for the emitted server bundle: its default export mirrors the
        // real build's shape (`default.default.fetch`), so `_ssr/ssr.mjs` must peel
        // the interop layers to find the Web fetch handler.
        fs::write(
            server_dir.join("server.mjs"),
            "const fetch = async (request) => {\n\
             \tconst { pathname } = new URL(request.url);\n\
             \tif (pathname === '/hello') return new Response('SSR-BODY-OK', { status: 200, headers: { 'content-type': 'text/html' } });\n\
             \treturn new Response('missing', { status: 404, headers: { 'content-type': 'text/html' } });\n\
             };\n\
             export default { default: { fetch } };\n",
        )
        .unwrap();
        // The natively generated manifest module: a runtime-style default export
        // carrying the `tsrStartManifest` factory that `_ssr/router.mjs` unwraps.
        fs::write(
            server_dir.join("_tanstack-start-manifest_v.mjs"),
            "const tsrStartManifest = () => ({ routes: { __root__: { preloads: [] } } });\n\
             export default { tsrStartManifest };\n",
        )
        .unwrap();
        fs::write(public_dir.join("static.txt"), "STATIC-ASSET-OK").unwrap();

        write_server_runtime_entry(&server_dir, false).unwrap();
        assert!(server_dir.join("index.mjs").is_file());
        assert!(server_dir.join("_ssr/ssr.mjs").is_file());
        assert!(server_dir.join("_ssr/router.mjs").is_file());
        assert!(server_dir.join("_ssr/node-adapter.mjs").is_file());

        // Reserve a free port, then hand it to the booted server.
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let mut child = node_command()
            .arg(server_dir.join("index.mjs"))
            .env("PORT", port.to_string())
            .env("HOST", "127.0.0.1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();

        let ssr = http_get_when_ready(port, "/hello");
        let asset = http_get_when_ready(port, "/static.txt");
        child.kill().ok();
        child.wait().ok();

        assert!(
            ssr.contains("200") && ssr.contains("SSR-BODY-OK"),
            "SSR response did not come from the handler: {ssr}"
        );
        assert!(
            asset.contains("200") && asset.contains("STATIC-ASSET-OK"),
            "static asset was not served from public/: {asset}"
        );
    }

    /// A minimal TanStack-style route app: a stub `@tanstack/react-router` (so no
    /// node_modules is needed), one route file with a split component, and an
    /// entry that imports it. Returns `(directory, entry, config)`.
    fn route_app_fixture() -> (tempfile::TempDir, PathBuf, BuildConfig) {
        let directory = tempdir().unwrap();
        let router_stub = directory.path().join("react-router.js");
        fs::write(
            &router_stub,
            "export const createFileRoute = () => (options) => options;\n\
             export const lazyRouteComponent = () => {};\n",
        )
        .unwrap();
        let routes = directory.path().join("routes");
        fs::create_dir(&routes).unwrap();
        fs::write(
            routes.join("foo.tsx"),
            "import { createFileRoute } from '@tanstack/react-router'\n\
             export const Route = createFileRoute('/foo')({\n  component: Foo,\n})\n\
             function Foo() {\n  return null\n}\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        fs::write(&entry, "import './routes/foo.tsx';\n").unwrap();

        let config = BuildConfig {
            base: "/".to_string(),
            browser_process_shim: false,
            asset_inline_limit: 0,
            aliases: vec![(
                "@tanstack/react-router".to_string(),
                router_stub.to_string_lossy().into_owned(),
            )],
            conditions: Vec::new(),
            main_fields: Vec::new(),
            virtual_modules: Vec::new(),
            target: Target::Server,
            server_external_packages: Vec::new(),
            import_meta_env: None,
            import_meta_glob: None,
            defines: Vec::new(),
            hmr: false,
            scss: crate::sass::ScssOptions::default(),
            image_import_shape: ImageImportShape::Url,
            css_preprocess: CssPreprocess::default(),
            jsx_extensions: crate::parser::JsxExtensions::JsxAndTsxOnly,
            jsx: crate::transform::JsxConfig::default(),
                    source_maps: false,
        };
        (directory, entry, config)
    }

    /// An app that imports ONE name (`publicValue`) from a `sideEffects:false`
    /// package whose other export wraps a value from a second `sideEffects:false`
    /// package in `createServerOnlyFn`. That second package (`@leaf/server`) is
    /// reachable only through the wrapper's reference to it — exactly the shape of
    /// the real `@tanstack/*` leak, where a bare-specifier `sideEffects:false`
    /// package carries the server-only `node:async_hooks` code. Returns
    /// `(directory, entry)`.
    fn server_leak_fixture() -> (tempfile::TempDir, PathBuf) {
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::write(
            root.join("package.json"),
            r#"{"name":"leak-app","version":"0.0.0"}"#,
        )
        .unwrap();
        let package = |name: &str, module_source: &str| {
            let dir = root.join("node_modules").join(name);
            fs::create_dir_all(&dir).unwrap();
            fs::write(
                dir.join("package.json"),
                format!(
                    r#"{{"name":"{name}","version":"0.0.0","module":"index.js","sideEffects":false}}"#
                ),
            )
            .unwrap();
            fs::write(dir.join("index.js"), module_source).unwrap();
        };
        // The directive-helper stub.
        package(
            "@tanstack/start-fn-stubs",
            "export const createServerOnlyFn = (fn) => fn;\n",
        );
        // The server-only leaf package (stands in for start-storage-context).
        package("@leaf/server", "export const serverThing = \"SERVER_ONLY_MARKER_9271\";\n");
        // The `sideEffects:false` barrel importing one name from each.
        package(
            "@tanstack/core",
            "import { createServerOnlyFn } from \"@tanstack/start-fn-stubs\";\n\
             import { serverThing } from \"@leaf/server\";\n\
             export const getServerThing = createServerOnlyFn(() => serverThing);\n\
             export const publicValue = 42;\n",
        );
        let entry = root.join("entry.js");
        fs::write(
            &entry,
            "import { publicValue } from \"@tanstack/core\";\nconsole.log(publicValue);\n",
        )
        .unwrap();
        (directory, entry)
    }

    #[test]
    fn client_build_drops_server_only_package_reached_through_neutralized_wrapper() {
        let (_directory, entry) = server_leak_fixture();
        let config = |target| BuildConfig {
            base: "/".to_string(),
            browser_process_shim: false,
            asset_inline_limit: 0,
            aliases: Vec::new(),
            conditions: Vec::new(),
            main_fields: Vec::new(),
            virtual_modules: Vec::new(),
            target,
            server_external_packages: Vec::new(),
            import_meta_env: None,
            import_meta_glob: None,
            defines: Vec::new(),
            hmr: false,
            scss: crate::sass::ScssOptions::default(),
            image_import_shape: ImageImportShape::Url,
            css_preprocess: CssPreprocess::default(),
            jsx_extensions: crate::parser::JsxExtensions::JsxAndTsxOnly,
            jsx: crate::transform::JsxConfig::default(),
                    source_maps: false,
        };

        // Client: `createServerOnlyFn(() => serverThing)` is neutralized to a
        // throwing stub, so `@leaf/server` is unreferenced and pruned by the
        // `sideEffects:false` shaking — the leaf never enters the client graph.
        let (client, _) =
            Bundler::discover_direct_with_config(&entry, &config(Target::Client)).unwrap();
        let client_reachable = client.reachable_modules_direct();
        assert!(
            !client_reachable
                .iter()
                .any(|module| module.contains("@leaf/server")),
            "the server-only package must not be reachable in the client build: {client_reachable:?}"
        );

        // Server: no transform, the wrapper keeps its reference, so the leaf stays.
        let (server, _) =
            Bundler::discover_direct_with_config(&entry, &config(Target::Server)).unwrap();
        let server_reachable = server.reachable_modules_direct();
        assert!(
            server_reachable
                .iter()
                .any(|module| module.contains("@leaf/server")),
            "the server-only package must remain reachable in the server build: {server_reachable:?}"
        );
    }

    #[test]
    fn client_route_manifest_attributes_split_chunks_to_route_ids() {
        let (_directory, entry, config) = route_app_fixture();
        let (bundler, update) = Bundler::discover_direct_with_config(&entry, &config).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();

        let manifest = bundler
            .client_route_manifest(&reachable, "client.js", "/")
            .unwrap();
        // The root route maps to the entry chunk (which statically bundles it).
        assert_eq!(
            manifest.routes.get(crate::manifest::ROOT_ROUTE_ID),
            Some(&vec!["client.js".to_string()])
        );
        // The route's split component becomes a dynamic chunk attributed to its
        // TanStack route id.
        let foo = manifest.routes.get("/foo").expect("route /foo is mapped");
        assert_eq!(foo.len(), 1, "one split chunk for /foo: {foo:?}");
        assert!(foo[0].starts_with("client.chunk-"), "{foo:?}");

        // The generated manifest source is the exact contract the server consumes.
        let source = manifest.to_start_manifest_source();
        assert!(source.contains("const tsrStartManifest = () => ({ clientEntry: \"/client.js\", routes: {"), "{source}");
        assert!(source.contains(&format!("\"/foo\": {{ preloads: [\"/{}\"] }}", foo[0])), "{source}");
    }

    #[test]
    fn a_registered_virtual_module_resolves_loads_and_names_its_chunk() {
        let directory = tempdir().unwrap();
        let entry = directory.path().join("server.ts");
        fs::write(
            &entry,
            "import('tanstack-start-manifest:v').then(({ tsrStartManifest }) => \
             console.log(tsrStartManifest()));\n",
        )
        .unwrap();

        let source =
            "const tsrStartManifest = () => ({ routes: {} });\nexport { tsrStartManifest };\n";
        let config = BuildConfig {
            base: "/".to_string(),
            browser_process_shim: false,
            asset_inline_limit: 0,
            aliases: Vec::new(),
            conditions: Vec::new(),
            main_fields: Vec::new(),
            virtual_modules: vec![(
                crate::manifest::START_MANIFEST_SPECIFIER.to_string(),
                source.to_string(),
            )],
            target: Target::Server,
            server_external_packages: Vec::new(),
            import_meta_env: None,
            import_meta_glob: None,
            defines: Vec::new(),
            hmr: false,
            scss: crate::sass::ScssOptions::default(),
            image_import_shape: ImageImportShape::Url,
            css_preprocess: CssPreprocess::default(),
            jsx_extensions: crate::parser::JsxExtensions::JsxAndTsxOnly,
            jsx: crate::transform::JsxConfig::default(),
                    source_maps: false,
        };
        let (bundler, update) = Bundler::discover_direct_with_config(&entry, &config).unwrap();
        // The previously-unresolvable specifier now resolves and loads: no gap.
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        assert!(
            bundler
                .all_modules()
                .contains(crate::manifest::START_MANIFEST_SPECIFIER),
            "the virtual module is in the graph"
        );

        let reachable = bundler.reachable_modules_direct();
        let output_root = directory.path().join(".diffpack-output");
        bundler
            .emit_server(&reachable, &output_root, EmitOptions::default())
            .unwrap();

        // The manifest lands in its own descriptively named server chunk (the
        // acceptance gate matches server files containing `tanstack-start-manifest`).
        let manifest_chunk = output_root.join("server/_tanstack-start-manifest_v.mjs");
        assert!(manifest_chunk.is_file(), "manifest chunk is emitted");
        let emitted = fs::read_to_string(&manifest_chunk).unwrap();
        assert!(emitted.contains("tsrStartManifest"), "{emitted}");
        node_check(&manifest_chunk);
    }

    /// Writes a `sideEffects`-annotated package under `<root>/node_modules/<name>`.
    /// `files` is `(relative path, source)`; `side_effects` is the raw JSON value
    /// of the `package.json` `sideEffects` field (e.g. `"false"`, `"true"`,
    /// `r#"["*.css"]"#`).
    fn write_package(root: &Path, name: &str, side_effects: &str, files: &[(&str, &str)]) {
        let package = root.join("node_modules").join(name);
        fs::create_dir_all(&package).unwrap();
        fs::write(
            package.join("package.json"),
            format!(
                "{{ \"name\": \"{name}\", \"version\": \"1.0.0\", \"main\": \"index.js\", \
                 \"sideEffects\": {side_effects} }}"
            ),
        )
        .unwrap();
        for (relative, source) in files {
            let path = package.join(relative);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(path, source).unwrap();
        }
    }

    #[test]
    fn dce_drops_a_barrel_reexported_module_no_live_module_uses() {
        // A `sideEffects:false` package whose barrel re-exports two modules; the
        // app uses only one. The unused re-exported module — and the
        // side-effectful module it pulls (which imports a Node built-in) — must be
        // dropped, exactly as Rollup/esbuild would.
        let directory = tempdir().unwrap();
        let root = directory.path();
        write_package(
            root,
            "lib",
            "false",
            &[
                (
                    "index.js",
                    "export { used } from './used.js';\nexport { unused } from './unused.js';\n",
                ),
                ("used.js", "export const used = 'USED';\n"),
                (
                    "unused.js",
                    "import { AsyncLocalStorage } from 'node:async_hooks';\n\
                     const store = new AsyncLocalStorage();\n\
                     export const unused = store;\n",
                ),
            ],
        );
        fs::write(root.join("package.json"), r#"{ "name": "app" }"#).unwrap();
        let entry = root.join("entry.js");
        fs::write(
            &entry,
            "import { used } from 'lib';\nconsole.log(used);\n",
        )
        .unwrap();

        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let live = bundler.live_modules(&reachable);

        let contains = |set: &BTreeSet<String>, suffix: &str| {
            set.iter().any(|id| id.ends_with(suffix))
        };
        // The barrel is reachable AND remains reachable, but `unused.js` is dead.
        assert!(contains(&reachable, "lib/unused.js"), "reachable set: {reachable:?}");
        assert!(
            !contains(&live, "lib/unused.js"),
            "the barrel-only, unused re-export must be dropped: {live:?}"
        );
        assert!(contains(&live, "lib/used.js"), "the used export must be kept: {live:?}");
        assert!(contains(&live, "lib/index.js"), "the live barrel is kept: {live:?}");

        // Emit and confirm the Node built-in the dead module pulled never ships.
        let output = root.join("dist/bundle.js");
        bundler.emit(&reachable, &output).unwrap();
        let bundle = fs::read_to_string(&output).unwrap();
        assert!(
            !bundle.contains("node:async_hooks"),
            "the dropped module's Node built-in must not ship: {bundle}"
        );
        assert!(bundle.contains("USED"), "the used export must ship: {bundle}");
        node_check(&output);
    }

    /// A module reached ONLY through a CommonJS `require()` must survive dead-module
    /// elimination.
    ///
    /// `sideEffects: false` authorizes dropping a module nothing demands, and demand was
    /// collected from `import` declarations alone — so a `require()`d module carried no
    /// demand whatsoever and was deleted. The `require` CALL survived, found nothing in
    /// the registry, and fell through to the external path: `MODULE_NOT_FOUND` under
    /// Node, and in the browser `Cannot require "…": it is not a Node built-in and was
    /// not included in the bundle`. That is exactly what killed hydration on every
    /// cal.com page, through
    /// `const { i18n } = require("@calcom/i18n/next-i18next.config")` in a
    /// `"sideEffects": false` workspace package.
    ///
    /// `require()` yields the whole `module.exports`, so the demand it places is the
    /// full namespace — there is no named subset to narrow it to.
    #[test]
    fn dce_keeps_a_module_reached_only_through_a_commonjs_require() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let root = directory.path();
        write_package(
            root,
            "config-pkg",
            "false",
            &[
                ("index.js", "module.exports = { unrelated: true };\n"),
                (
                    "settings.js",
                    "module.exports = { locales: ['en', 'fr'] };\n",
                ),
            ],
        );
        fs::write(root.join("package.json"), r#"{ "name": "app" }"#).unwrap();
        // The require sits in a module that ALSO has ESM structure, so the liveness
        // record is non-empty and the conservative "no captured structure" path (which
        // keeps every dependency) cannot be what saves it.
        fs::write(
            root.join("lib.js"),
            "const settings = require('config-pkg/settings.js');\n\
             export const locales = settings.locales;\n",
        )
        .unwrap();
        let entry = root.join("entry.js");
        fs::write(
            &entry,
            "import { locales } from './lib.js';\nconsole.log('locales:' + locales.join(','));\n",
        )
        .unwrap();

        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let live = bundler.live_modules(&reachable);
        assert!(
            live.iter().any(|id| id.ends_with("config-pkg/settings.js")),
            "a require()d module must stay live even under sideEffects:false: {live:?}"
        );
        // The unrelated entry point of the same package is still droppable — the fix
        // must not degrade into "keep the whole package".
        assert!(
            !live.iter().any(|id| id.ends_with("config-pkg/index.js")),
            "only what is actually required is kept: {live:?}"
        );

        // Executing is the real assertion: the emitted `require` must find its target
        // in the registry rather than falling through to the host.
        let output = root.join("dist/bundle.mjs");
        bundler
            .emit_with_options(
                &reachable,
                &output,
                EmitOptions {
                    format: ModuleFormat::Esm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();
        assert_eq!(run_node(&output), "locales:en,fr\n");
    }

    #[test]
    fn dce_keeps_a_side_effectful_module_and_a_used_module() {
        // Two packages: one `sideEffects:true` (its module runs for effect even if
        // nothing is imported from it) and one `sideEffects:false` whose export IS
        // used. Both must be kept.
        let directory = tempdir().unwrap();
        let root = directory.path();
        write_package(
            root,
            "effectful",
            "true",
            &[("index.js", "globalThis.__EFFECT__ = true;\n")],
        );
        write_package(
            root,
            "pure",
            "false",
            &[("index.js", "export const value = 'PURE';\n")],
        );
        fs::write(root.join("package.json"), r#"{ "name": "app" }"#).unwrap();
        let entry = root.join("entry.js");
        fs::write(
            &entry,
            "import 'effectful';\nimport { value } from 'pure';\nconsole.log(value);\n",
        )
        .unwrap();

        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let live = bundler.live_modules(&reachable);

        let contains = |set: &BTreeSet<String>, suffix: &str| {
            set.iter().any(|id| id.ends_with(suffix))
        };
        assert!(
            contains(&live, "effectful/index.js"),
            "a bare `import 'effectful'` of a sideEffects:true module must be kept: {live:?}"
        );
        assert!(
            contains(&live, "pure/index.js"),
            "a used sideEffects:false module must be kept: {live:?}"
        );
    }

    #[test]
    fn dce_drops_a_bare_side_effect_import_of_a_side_effect_free_module() {
        // `import './noop.js'` for effect, but `./noop.js`'s package declares
        // `sideEffects:false`, so the flag authorizes dropping the module (and its
        // Node-built-in import) entirely — matching Rollup/esbuild.
        let directory = tempdir().unwrap();
        let root = directory.path();
        write_package(
            root,
            "quiet",
            "false",
            &[
                ("index.js", "export const marker = 'QUIET';\n"),
                (
                    "noop.js",
                    "import { readFileSync } from 'node:fs';\nexport const noop = readFileSync;\n",
                ),
            ],
        );
        fs::write(root.join("package.json"), r#"{ "name": "app" }"#).unwrap();
        let entry = root.join("entry.js");
        // Import the package's `noop.js` purely for side effect.
        fs::write(&entry, "import 'quiet/noop.js';\nconsole.log('app');\n").unwrap();

        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let live = bundler.live_modules(&reachable);
        let contains = |set: &BTreeSet<String>, suffix: &str| {
            set.iter().any(|id| id.ends_with(suffix))
        };
        assert!(
            !contains(&live, "quiet/noop.js"),
            "a bare side-effect import of a sideEffects:false module must be droppable: {live:?}"
        );

        let output = root.join("dist/bundle.js");
        bundler.emit(&reachable, &output).unwrap();
        let bundle = fs::read_to_string(&output).unwrap();
        assert!(
            !bundle.contains("node:fs"),
            "the dropped side-effect module's Node built-in must not ship: {bundle}"
        );
        node_check(&output);
    }

    /// A build that opts into Vite conventions for `import.meta.glob`, rooted at
    /// `root` (the gate `config::derive_web_config --vite` and `build-app` set).
    fn glob_config(root: &Path) -> BuildConfig {
        BuildConfig {
            base: "/".to_string(),
            browser_process_shim: false,
            asset_inline_limit: 0,
            import_meta_glob: Some(crate::import_meta_glob::ImportMetaGlob {
                root: root.canonicalize().unwrap(),
            }),
            ..BuildConfig::default()
        }
    }

    fn emitted_chunk_names(dist: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dist)
            .unwrap()
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.starts_with("bundle.chunk-"))
            .collect();
        names.sort();
        names
    }

    #[test]
    fn import_meta_glob_lazy_matches_load_from_their_own_chunks_in_sorted_key_order() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let widgets = directory.path().join("widgets");
        fs::create_dir_all(&widgets).unwrap();
        // Written in reverse name order so sorted keys are the transform's doing.
        fs::write(widgets.join("beta.js"), "export const name = 'beta';\n").unwrap();
        fs::write(widgets.join("alpha.js"), "export const name = 'alpha';\n").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "const modules = import.meta.glob('./widgets/*.js');\n\
             console.log(JSON.stringify(Object.keys(modules)));\n\
             Promise.all(Object.entries(modules).map(async ([key, load]) => `${key}=${(await load()).name}`))\n\
               .then((loaded) => console.log(loaded.join(',')));\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) =
            Bundler::discover_direct_with_config(&entry, &glob_config(directory.path())).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();

        // Each lazy match is its own dynamic-import graph edge, so its own chunk.
        let chunks = emitted_chunk_names(&directory.path().join("dist"));
        assert_eq!(chunks.len(), 2, "one chunk per lazy match: {chunks:?}");

        let executed = node_command().arg(&output).output().unwrap();
        assert!(
            executed.status.success(),
            "{}",
            String::from_utf8_lossy(&executed.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&executed.stdout),
            "[\"./widgets/alpha.js\",\"./widgets/beta.js\"]\n\
             ./widgets/alpha.js=alpha,./widgets/beta.js=beta\n"
        );
    }

    #[test]
    fn import_meta_glob_eager_with_default_import_binds_values_statically() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let widgets = directory.path().join("widgets");
        fs::create_dir_all(&widgets).unwrap();
        fs::write(widgets.join("alpha.js"), "export default 'A';\n").unwrap();
        fs::write(widgets.join("beta.js"), "export default 'B';\n").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "const modules = import.meta.glob('./widgets/*.js', { eager: true, import: 'default' });\n\
             console.log(modules['./widgets/alpha.js'], modules['./widgets/beta.js']);\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) =
            Bundler::discover_direct_with_config(&entry, &glob_config(directory.path())).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();

        // Eager matches are static imports: everything lands in the entry chunk.
        let chunks = emitted_chunk_names(&directory.path().join("dist"));
        assert!(chunks.is_empty(), "eager glob must not split chunks: {chunks:?}");

        let executed = node_command().arg(&output).output().unwrap();
        assert!(
            executed.status.success(),
            "{}",
            String::from_utf8_lossy(&executed.stderr)
        );
        assert_eq!(String::from_utf8_lossy(&executed.stdout), "A B\n");
    }

    #[test]
    fn import_meta_glob_raw_query_routes_matches_through_the_raw_loader() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let notes = directory.path().join("notes");
        fs::create_dir_all(&notes).unwrap();
        fs::write(notes.join("hello.txt"), "hello from glob raw").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "const files = import.meta.glob('./notes/*.txt', { eager: true, import: 'default', query: '?raw' });\n\
             console.log(files['./notes/hello.txt']);\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) =
            Bundler::discover_direct_with_config(&entry, &glob_config(directory.path())).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();

        let executed = node_command().arg(&output).output().unwrap();
        assert!(
            executed.status.success(),
            "{}",
            String::from_utf8_lossy(&executed.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&executed.stdout),
            "hello from glob raw\n"
        );
    }

    #[test]
    fn import_meta_glob_pattern_array_unions_and_negative_pattern_excludes() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::create_dir_all(directory.path().join("a")).unwrap();
        fs::create_dir_all(directory.path().join("b")).unwrap();
        fs::write(directory.path().join("a/one.js"), "export const v = 1;\n").unwrap();
        fs::write(directory.path().join("a/skip.js"), "export const v = 0;\n").unwrap();
        fs::write(directory.path().join("b/two.js"), "export const v = 2;\n").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "const modules = import.meta.glob(['./a/*.js', './b/*.js', '!**/skip.js']);\n\
             console.log(JSON.stringify(Object.keys(modules)));\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) =
            Bundler::discover_direct_with_config(&entry, &glob_config(directory.path())).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();

        let executed = node_command().arg(&output).output().unwrap();
        assert!(
            executed.status.success(),
            "{}",
            String::from_utf8_lossy(&executed.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&executed.stdout),
            "[\"./a/one.js\",\"./b/two.js\"]\n"
        );
    }

    #[test]
    fn without_the_vite_opt_in_import_meta_glob_is_left_untouched() {
        let directory = tempdir().unwrap();
        let widgets = directory.path().join("widgets");
        fs::create_dir_all(&widgets).unwrap();
        fs::write(widgets.join("alpha.js"), "export const name = 'alpha';\n").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "export const modules = import.meta.glob('./widgets/*.js');\n",
        )
        .unwrap();

        // No `import_meta_glob` in the config: generic bundling. The call must
        // survive to the module (no expansion, no graph edges), so the existing
        // import.meta-in-CommonJS honesty check refuses the CJS emit by name.
        let entry = directory.path().join("entry.js");
        let (bundler, update) = Bundler::discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        assert_eq!(reachable.len(), 1, "no glob edges without the opt-in: {reachable:?}");
        let error = bundler
            .emit(&reachable, &directory.path().join("dist/bundle.js"))
            .unwrap_err();
        assert!(error.contains("import.meta"), "{error}");
        assert!(error.contains("entry.js"), "{error}");
    }

    #[test]
    fn asset_variant_public_name_appends_width_before_ext() {
        assert_eq!(
            asset_variant_public_name("shot-1a2b3c4d.png", 640),
            "shot-1a2b3c4d-640.png"
        );
        assert_eq!(asset_variant_public_name("noext", 32), "noext-32");
    }

    #[test]
    fn blur_data_url_is_a_tiny_decodable_data_uri() {
        let img = image::DynamicImage::new_rgb8(200, 100);
        let png = generate_blur_data_url(&img, "png").unwrap();
        assert!(png.starts_with("data:image/png;base64,"), "{png}");
        let jpeg = generate_blur_data_url(&img, "jpeg").unwrap();
        assert!(jpeg.starts_with("data:image/jpeg;base64,"), "{jpeg}");
        // A real payload but small (~8px-wide downscale), never a heavy full image.
        assert!(png.len() > 40 && png.len() < 4000, "tiny but real: {}", png.len());
    }

    #[test]
    fn next_object_image_import_differs_from_vite_url_shape() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("shot.png");
        image::DynamicImage::new_rgb8(300, 200).save(&path).unwrap();

        // NextObject: the module plans responsive variants to emit (object shape).
        let obj = synthesize_asset_url(
            path.clone(),
            "/",
            0,
            ImageImportShape::NextObject { responsive_variants: true },
        )
        .unwrap();
        assert_eq!(obj.assets.len(), 1, "one emitted original");
        let variants = obj.assets[0]
            .image_variants
            .as_ref()
            .expect("NextObject plans responsive variants");
        assert_eq!(variants, &crate::next_adapter::variant_widths(300));
        assert!(variants.len() >= 2, "several responsive widths: {variants:?}");
        assert!(obj.code.contains("variants"), "the object carries its ladder: {}", obj.code);

        // NextObject with optimization off (next.config `images.unoptimized` / a custom
        // loader): the SAME object shape — Next's static import always carries
        // src/width/height/blurDataURL — but no ladder is planned and the `variants`
        // key is OMITTED, which is what makes the shim render a raw <img src>. An empty
        // `{}` would be truthy and would silently keep the srcset path alive.
        let unopt = synthesize_asset_url(
            path.clone(),
            "/",
            0,
            ImageImportShape::NextObject { responsive_variants: false },
        )
        .unwrap();
        assert_eq!(unopt.assets.len(), 1, "the original is still emitted");
        assert!(
            unopt.assets[0].image_variants.is_none(),
            "no variant file is planned when optimization is off",
        );
        assert!(unopt.code.contains("blurDataURL"), "blur is still generated: {}", unopt.code);
        assert!(unopt.code.contains("width"), "intrinsic size is still carried: {}", unopt.code);
        assert!(
            !unopt.code.contains("variants"),
            "the `variants` key is omitted, not emptied: {}",
            unopt.code,
        );

        // Url (Vite/TanStack/generic): bare URL string, NO variants planned. This
        // locks the no-regression guarantee for every non-Next build path.
        let url = synthesize_asset_url(path.clone(), "/", 0, ImageImportShape::Url).unwrap();
        assert_eq!(url.assets.len(), 1);
        assert!(
            url.assets[0].image_variants.is_none(),
            "Url mode stays bare-URL (Vite parity): no variants"
        );
    }

    // --- diagnostic fatality ------------------------------------------------
    //
    // The predicate that decides whether a build fails. It is structural (the
    // diagnostic's kind), not a substring match, so a new diagnostic kind has to
    // state its own fatality rather than inherit someone else's.

    #[test]
    fn an_unresolved_import_is_a_fatal_diagnostic() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { helper } from './does-not-exist.js';\nexport const value = helper;\n",
        )
        .unwrap();

        let (_, update) = Bundler::discover_direct(&directory.path().join("entry.js")).unwrap();
        assert_eq!(update.diagnostics.len(), 1, "{:?}", update.diagnostics);
        let diagnostic = &update.diagnostics[0];
        assert!(matches!(
            &diagnostic.kind,
            DiagnosticKind::UnresolvedImport { specifier, .. }
                if specifier == "./does-not-exist.js"
        ));
        assert!(diagnostic.is_fatal());
        // The message must be actionable: it names the specifier, the importing
        // file, and (for a relative path) that no file matched.
        assert!(diagnostic.message.contains("./does-not-exist.js"));
        assert!(diagnostic.message.contains("entry.js"));
        assert!(diagnostic.message.contains("no file matched"));

        let error = partition_diagnostics(&update.diagnostics, "test build").unwrap_err();
        assert!(error.contains("test build"), "{error}");
        assert!(error.contains("./does-not-exist.js"), "{error}");
    }

    #[test]
    fn an_unresolved_bare_package_suggests_installing_it() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import x from '@scope/missing-pkg/sub';\nexport const value = x;\n",
        )
        .unwrap();

        let (_, update) = Bundler::discover_direct(&directory.path().join("entry.js")).unwrap();
        assert_eq!(update.diagnostics.len(), 1, "{:?}", update.diagnostics);
        assert!(
            update.diagnostics[0]
                .message
                .contains("npm install @scope/missing-pkg"),
            "{}",
            update.diagnostics[0].message
        );
    }

    #[test]
    fn a_node_builtin_is_an_external_not_a_diagnostic() {
        // Locks in that making unresolved imports fatal can never start failing
        // builds over Node built-ins: they short-circuit before resolution and are
        // never diagnostics at all.
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { readFileSync } from 'node:fs';\nimport { join } from 'path';\n\
             export const read = (p) => readFileSync(join(p, 'x'));\n",
        )
        .unwrap();

        let config = BuildConfig {
            target: Target::Server,
            ..BuildConfig::default()
        };
        let (bundler, update) =
            Bundler::discover_direct_with_config(&directory.path().join("entry.js"), &config)
                .unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        assert!(partition_diagnostics(&update.diagnostics, "test build").is_ok());
        let reachable = bundler.reachable_modules_direct();
        assert_eq!(reachable.len(), 1);
    }

    #[test]
    fn an_unsupported_side_effects_glob_is_a_warning_and_the_build_succeeds() {
        // `"sideEffects": ["*.{css,scss}"]` is a common package.json idiom this
        // matcher cannot evaluate. The module is KEPT, so the bundle is correct —
        // only larger. Failing the build on it would reject apps that bundle fine.
        let directory = tempdir().unwrap();
        let package = directory.path().join("node_modules/braced-pkg");
        fs::create_dir_all(&package).unwrap();
        fs::write(
            package.join("package.json"),
            r#"{"name":"braced-pkg","type":"module","exports":"./index.js","sideEffects":["*.{css,scss}"]}"#,
        )
        .unwrap();
        fs::write(package.join("index.js"), "export const value = 'ok';").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { value } from 'braced-pkg';\nconsole.log(value);\n",
        )
        .unwrap();

        let (bundler, update) = Bundler::discover_direct(&directory.path().join("entry.js")).unwrap();
        let side_effects = update
            .diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.kind == DiagnosticKind::SideEffectsGlob)
            .collect::<Vec<_>>();
        assert_eq!(side_effects.len(), 1, "{:?}", update.diagnostics);
        assert!(!side_effects[0].is_fatal());

        let warnings = partition_diagnostics(&update.diagnostics, "test build")
            .expect("an unsupported sideEffects glob must not fail the build");
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("sideEffects"), "{}", warnings[0]);
        // And the build really does complete.
        let output = directory.path().join("dist/bundle.js");
        bundler
            .emit(&bundler.reachable_modules_direct(), &output)
            .unwrap();
        assert!(output.is_file());
    }

    #[test]
    fn partition_diagnostics_reports_every_fatal_and_keeps_warnings_separate() {
        let diagnostics = vec![
            Diagnostic {
                kind: DiagnosticKind::UnresolvedImport {
                    specifier: "./a".into(),
                    importer: PathBuf::from("/app/one.js"),
                },
                message: "cannot resolve \"./a\"".into(),
            },
            Diagnostic {
                kind: DiagnosticKind::SideEffectsGlob,
                message: "unsupported `sideEffects` glob".into(),
            },
            Diagnostic {
                kind: DiagnosticKind::Source { fatal: false },
                message: "a benign oxc warning".into(),
            },
            Diagnostic {
                kind: DiagnosticKind::Source { fatal: true },
                message: "a real parse error".into(),
            },
        ];

        let error = partition_diagnostics(&diagnostics, "client build").unwrap_err();
        assert!(error.contains("2 fatal build diagnostic(s)"), "{error}");
        assert!(error.contains("cannot resolve \"./a\""), "{error}");
        assert!(error.contains("a real parse error"), "{error}");
        assert!(!error.contains("a benign oxc warning"), "{error}");

        let warnings = partition_diagnostics(&diagnostics[1..3], "client build").unwrap();
        assert_eq!(
            warnings,
            vec![
                "unsupported `sideEffects` glob".to_string(),
                "a benign oxc warning".to_string()
            ]
        );
    }

    /// FINDINGS #19. A legacy v3 app's design tokens are `theme.extend` ON TOP OF the
    /// v3 DEFAULT theme — a different palette, radius scale and type scale from v4's.
    /// The evaluator used to emit only the config's OWN keys, which diffpack then
    /// merged into the vendored v4 defaults, so every unmentioned token came out v4:
    /// `slate-400` as `oklch(...)` rather than `#94a3b8`, `rounded-full` as
    /// `calc(infinity * 1px)` rather than `9999px`. It now resolves the config through
    /// the app's own `tailwindcss/resolveConfig`.
    ///
    /// Runs against the pinned `next-blog-starter` e2e app (a real tailwindcss@3
    /// install); soft-skips when the corpus has not been fetched.
    #[test]
    fn v3_config_evaluator_resolves_the_full_v3_default_theme() {
        let app = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("integration/e2e/apps/next-blog-starter");
        let config = app.join("tailwind.config.ts");
        if !config.is_file() || !app.join("node_modules/tailwindcss").is_dir() {
            return;
        }
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let loader = std::env::temp_dir().join("diffpack-tailwind-config-eval-test.mjs");
        fs::write(&loader, include_str!("../scripts/tailwind-config-eval.mjs")).unwrap();
        let output = node_command()
            .arg(&loader)
            .arg(&config)
            .current_dir(&app)
            .output()
            .unwrap();
        let theme = String::from_utf8_lossy(&output.stdout).to_string();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));

        // v3 DEFAULT tokens the config never mentions, in v3's own sRGB form.
        assert!(theme.contains("--color-slate-400: #94a3b8;"), "{theme}");
        assert!(theme.contains("--radius-full: 9999px;"), "{theme}");
        // The v3 preflight's border reset colour.
        assert!(theme.contains("--default-border-color: #e5e7eb;"), "{theme}");
        // A `[size, { lineHeight }]` pair splits into the value + modifier tokens
        // instead of stringifying the modifier object into the font-size.
        assert!(theme.contains("--text-4xl: 2.25rem;"), "{theme}");
        assert!(theme.contains("--text-4xl--line-height: 2.5rem;"), "{theme}");
        assert!(!theme.contains("[object Object]"), "{theme}");
        // The app's OWN tokens still win over the resolved defaults.
        assert!(theme.contains("--color-cyan: #79FFE1;"), "{theme}");
        assert!(theme.contains("--shadow-md: 0 8px 30px rgba(0, 0, 0, 0.12);"), "{theme}");
        // v3 `columns.12` is a column COUNT, not a v4 `--container-12` width: emitting
        // it made `w-12` resolve against the container scale (100px, not 3rem).
        assert!(!theme.contains("--container-12:"), "{theme}");
        assert!(theme.contains("--spacing-12: 3rem;"), "{theme}");

        // `darkMode: "class"` carries across as the `dark` variant it defines. Without
        // it every `dark:` utility compiled into `@media (prefers-color-scheme: dark)`,
        // so the app painted its dark palette on a browser that merely preferred dark.
        assert!(theme.contains("@custom-variant dark (&:is(.dark *));"), "{theme}");

        // The resolved v3 fontSize scale REPLACES the vendored v4 one. Merging left
        // v4's `--text-5xl--line-height: 1` in place, but this config sets
        // `fontSize: { '5xl': '2.5rem' }` — a bare string, i.e. no line-height at all.
        assert!(theme.contains("--text-*: initial;"), "{theme}");
        assert!(theme.contains("--text-5xl: 2.5rem;"), "{theme}");
        assert!(!theme.contains("--text-5xl--line-height"), "{theme}");
        // The reset comes first, so every size token after it survives.
        assert!(
            theme.find("--text-*: initial;").unwrap() < theme.find("--text-4xl: 2.25rem;").unwrap(),
            "{theme}"
        );
    }

    /// `darkMode` strategies the evaluator maps, and the hard error for one it does
    /// not: an untranslated strategy would silently fall back to the media query.
    #[test]
    fn v3_config_evaluator_maps_every_dark_mode_strategy() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let loader = std::env::temp_dir().join("diffpack-tailwind-darkmode-eval-test.mjs");
        fs::write(&loader, include_str!("../scripts/tailwind-config-eval.mjs")).unwrap();
        let dir = std::env::temp_dir().join("diffpack-tailwind-darkmode-configs");
        fs::create_dir_all(&dir).unwrap();
        let run = |name: &str, dark_mode: &str| {
            let config = dir.join(format!("{name}.cjs"));
            fs::write(&config, format!("module.exports = {{ {dark_mode} theme: {{}} }};\n")).unwrap();
            let out = node_command().arg(&loader).arg(&config).output().unwrap();
            (
                String::from_utf8_lossy(&out.stdout).to_string(),
                String::from_utf8_lossy(&out.stderr).to_string(),
                out.status.success(),
            )
        };

        let (media, _, ok) = run("media", "darkMode: 'media',");
        assert!(ok);
        assert!(!media.contains("@custom-variant"), "{media}");
        let (absent, _, ok) = run("absent", "");
        assert!(ok);
        assert!(!absent.contains("@custom-variant"), "{absent}");

        // v3's `class` strategy emits `<selector> &`; `selector` emits the
        // `:where(sel, sel *)` form that also matches the element itself.
        let (class, _, ok) = run("class", "darkMode: 'class',");
        assert!(ok);
        assert!(class.contains("@custom-variant dark (&:is(.dark *));"), "{class}");
        let (named, _, ok) = run("named", "darkMode: ['class', '[data-mode=\"dark\"]'],");
        assert!(ok);
        assert!(named.contains("@custom-variant dark (&:is([data-mode=\"dark\"] *));"), "{named}");
        let (selector, _, ok) = run("selector", "darkMode: 'selector',");
        assert!(ok);
        assert!(selector.contains("@custom-variant dark (&:where(.dark, .dark *));"), "{selector}");

        // An unmapped strategy is a hard, named failure — never a silent fallback.
        let (_, stderr, ok) = run("variant", "darkMode: ['variant', '&:not(.light *)'],");
        assert!(!ok, "an unmapped darkMode strategy must fail the evaluation");
        assert!(stderr.contains("darkMode"), "{stderr}");
    }
}
