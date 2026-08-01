//! Framework-independent chunk planning and rendered-output records.

use std::collections::{BTreeMap, VecDeque};
use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::Path;
use std::sync::Arc;

use oxc_allocator::Allocator;
use oxc_ast_visit::{Visit, walk};
use oxc_parser::Parser;
use oxc_span::{SourceType, Span};
use rayon::prelude::*;

use crate::async_graph::{AsyncModules, rewrite_imports as await_async_imports};
use crate::emission::ModuleFormat;
use crate::module_graph::{StaticGraphView, static_execution_order};
use crate::source_map::{
    LineTrack, MapToken, ModuleMapLookup, ModuleSourceMap, ResolvedMinifiedToken, is_identifier,
    line_count, replace_tracked,
};
use crate::transform::{DependencyDemand, FlatModule};
use crate::tree_shake::shake as shake_module_code;
use crate::tree_shake::{Demand, ExportDemand as _, shake};
use oxc_sourcemap::SourceMapBuilder;

pub type DenseModuleId = usize;

struct ImportMetaSpans(Vec<Span>);

impl<'a> Visit<'a> for ImportMetaSpans {
    fn visit_meta_property(&mut self, meta: &oxc_ast::ast::MetaProperty<'a>) {
        if meta.meta.name == "import" && meta.property.name == "meta" {
            self.0.push(meta.span);
        }
        walk::walk_meta_property(self, meta);
    }
}

fn lower_cjs_import_meta(code: &str, module_id: &str) -> Option<String> {
    let allocator = Allocator::default();
    let parsed = Parser::new(&allocator, code, SourceType::mjs()).parse();
    if parsed.panicked {
        return None;
    }
    let mut spans = ImportMetaSpans(Vec::new());
    spans.visit_program(&parsed.program);
    if spans.0.is_empty() {
        return None;
    }
    let mut output = code.to_string();
    for span in spans.0.into_iter().rev() {
        output.replace_range(
            span.start as usize..span.end as usize,
            "__diffpackImportMeta",
        );
    }
    let escaped_path = module_id
        .replace('%', "%25")
        .replace(' ', "%20")
        .replace('#', "%23")
        .replace('?', "%3F");
    Some(format!(
        "const __diffpackImportMeta={{url:{},env:Object.create(null)}};\n{output}",
        quote(&format!("file://{escaped_path}"))
    ))
}

#[derive(Debug, Clone)]
pub struct ChunkPlan {
    pub members: Vec<DenseModuleId>,
    pub roots: Vec<DenseModuleId>,
    pub prerequisites: Vec<usize>,
    pub file_name: String,
}

/// Framework-neutral view of an emitted graph for integration-owned manifests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegrationManifestGraph {
    pub entry_file: String,
    pub modules: Vec<IntegrationManifestModule>,
    pub chunks: Vec<IntegrationManifestChunk>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegrationManifestModule {
    pub id: String,
    pub source: String,
    pub runtime_id: usize,
    pub chunk: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegrationManifestChunk {
    pub roots: Vec<String>,
    /// Prerequisite closure followed by this chunk, in runtime load order.
    pub load_order: Vec<String>,
}

pub struct EmitPlan {
    pub reachable_dense: Vec<DenseModuleId>,
    pub allowed: HashSet<DenseModuleId>,
    pub runtime_ids: Vec<Option<usize>>,
    pub chunk_of: HashMap<DenseModuleId, String>,
    pub chunk_names: HashMap<DenseModuleId, String>,
    pub chunk_files: Vec<String>,
}

#[derive(Clone)]
pub struct RenderedBundle {
    pub code: String,
    pub mappings: Vec<ModuleMapping>,
    pub map_json: Option<String>,
}

#[derive(Clone)]
pub struct ModuleMapping {
    pub dense_index: DenseModuleId,
    pub generated_line: u32,
    pub tokens: Vec<MapToken>,
}

/// Borrowed module data required to render one registry-runtime factory.
pub struct RuntimeRenderModule<'a> {
    pub id: &'a str,
    pub code: &'a str,
    pub dependencies: &'a [(String, DenseModuleId, DependencyDemand)],
    pub pruned_imports: &'a HashSet<String>,
    pub map: Option<&'a ModuleSourceMap>,
    pub uses_dirname: bool,
}

/// One module's contribution to the registry literals and readable source map.
pub struct RuntimeFragment {
    pub dense_index: DenseModuleId,
    pub module: String,
    pub import_map: String,
    pub chunk_map: String,
    pub generated_lines: usize,
    pub track: Option<LineTrack>,
}

/// Format-specific text emitted before a registry chunk and its source-map
/// displacement. Framework and host integrations supply only their optional
/// entry preludes; core owns ordering and module-format syntax.
pub struct RuntimeHeader {
    pub prelude: String,
    pub prerequisite_loads: String,
    pub generated_lines: u32,
}

pub struct RuntimeLiterals {
    pub modules: String,
    pub import_maps: String,
    pub chunk_maps: String,
    pub mappings: Vec<ModuleMapping>,
}

/// Joins rendered factories into registry literals and projects each module's
/// real map through its render-time line edits onto final chunk coordinates.
pub fn assemble_runtime_literals<'a>(
    fragments: Vec<RuntimeFragment>,
    header_lines: u32,
    module_map: impl Fn(DenseModuleId) -> Option<&'a ModuleSourceMap>,
) -> RuntimeLiterals {
    let mut modules = String::new();
    let mut import_maps = String::new();
    let mut chunk_maps = String::new();
    let mut mappings = Vec::with_capacity(fragments.len());
    let mut module_lines = 0_u32;
    for fragment in fragments {
        let region_line = 3 + header_lines + module_lines;
        let mut tokens = Vec::new();
        if let (Some(track), Some(map)) =
            (fragment.track.as_ref(), module_map(fragment.dense_index))
        {
            track.project(map, region_line, &mut tokens);
        }
        mappings.push(ModuleMapping {
            dense_index: fragment.dense_index,
            generated_line: region_line,
            tokens,
        });
        module_lines += fragment.module.matches('\n').count() as u32;
        modules.push_str(&fragment.module);
        import_maps.push_str(&fragment.import_map);
        chunk_maps.push_str(&fragment.chunk_map);
    }
    RuntimeLiterals {
        modules,
        import_maps,
        chunk_maps,
        mappings,
    }
}

pub fn render_runtime_header(
    format: ModuleFormat,
    prerequisites: &[String],
    preludes: &[&str],
) -> RuntimeHeader {
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
    let mut prelude = String::new();
    for fragment in preludes {
        prelude.push_str(fragment);
    }
    let generated_lines =
        (prelude.matches('\n').count() + prerequisite_loads.matches('\n').count()) as u32;
    RuntimeHeader {
        prelude,
        prerequisite_loads,
        generated_lines,
    }
}

/// Renders the framework-neutral per-module portion of a registry bundle.
///
/// Runtime wrappers, HMR policy, browser host shims, and framework compatibility
/// preludes remain caller-supplied concerns. Cancellation returns `None`, never a
/// partial fragment set.
#[allow(clippy::too_many_arguments)]
pub fn render_runtime_fragments(
    modules: &[Option<RuntimeRenderModule<'_>>],
    reachable: &[DenseModuleId],
    roots: &[DenseModuleId],
    chunk_names: &HashMap<DenseModuleId, String>,
    runtime_ids: &[Option<usize>],
    global_demands: &[Demand],
    async_modules: &AsyncModules,
    format: ModuleFormat,
    cancelled: &(impl Fn() -> bool + Sync),
) -> Option<Vec<RuntimeFragment>> {
    let mut export_demands = global_demands.to_vec();
    for &root in roots {
        export_demands[root].all = true;
    }
    let stop = std::sync::atomic::AtomicBool::new(false);
    let fragments = reachable
        .par_iter()
        .filter_map(|&dense_index| {
            if stop.load(std::sync::atomic::Ordering::Relaxed) {
                return None;
            }
            if cancelled() {
                stop.store(true, std::sync::atomic::Ordering::Relaxed);
                return None;
            }
            let module = modules[dense_index].as_ref()?;
            let runtime_id = runtime_ids[dense_index]
                .expect("a rendered module must have a deterministic runtime ID");
            let mut pruned_imports = module.pruned_imports.clone();
            let mut dropped_targets = Vec::new();
            for (specifier, target, _) in module.dependencies {
                if runtime_ids[*target].is_none() {
                    pruned_imports.insert(specifier.clone());
                    dropped_targets.push(specifier.as_str());
                }
            }
            let is_async_module = async_modules.is_async(dense_index);
            let mut track = module
                .map
                .map(|_| LineTrack::identity(module.code.lines().count()));
            let mut lowered = std::borrow::Cow::Borrowed(module.code);
            for specifier in dropped_targets {
                let call = format!("__reExport(exports,require.esm({}));", quote(specifier));
                if lowered.contains(&call) {
                    let rewritten = lowered.replace(&call, "");
                    if let Some(track) = track.as_mut() {
                        track.invalidate_changed_lines(&lowered, &rewritten);
                    }
                    lowered = std::borrow::Cow::Owned(rewritten);
                }
            }
            if is_async_module {
                for (specifier, target, _) in module.dependencies {
                    if runtime_ids[*target].is_some() && async_modules.is_async(*target) {
                        let rewritten = await_async_imports(&lowered, specifier);
                        if let Some(track) = track.as_mut() {
                            track.invalidate_changed_lines(&lowered, &rewritten);
                        }
                        lowered = std::borrow::Cow::Owned(rewritten);
                    }
                }
            }
            if format == ModuleFormat::Cjs
                && let Some(rewritten) = lower_cjs_import_meta(&lowered, module.id)
            {
                lowered = std::borrow::Cow::Owned(rewritten);
                track = None;
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
            let mut import_map = format!("{runtime_id}:{{");
            let mut chunk_map = format!("{runtime_id}:{{");
            for (specifier, target, demand) in module.dependencies {
                let Some(target_runtime_id) = runtime_ids[*target] else {
                    continue;
                };
                import_map.push_str(&format!("{}:{target_runtime_id},", quote(specifier)));
                if demand.dynamic {
                    let chunk = chunk_names
                        .get(target)
                        .map_or("null".to_owned(), |chunk| quote(chunk));
                    chunk_map.push_str(&format!(
                        "{}:[{chunk},{target_runtime_id}],",
                        quote(specifier)
                    ));
                }
            }
            import_map.push_str("},\n");
            chunk_map.push_str("},\n");
            Some(RuntimeFragment {
                dense_index,
                module: module_fragment,
                import_map,
                chunk_map,
                generated_lines: code.lines().count(),
                track,
            })
        })
        .collect::<Vec<_>>();
    (!stop.load(std::sync::atomic::Ordering::Relaxed)).then_some(fragments)
}

/// Refuses readable chunk mappings that point outside the emitted UTF-16 text.
pub fn validate_mappings(
    code: &str,
    mappings: &[ModuleMapping],
    chunk_name: &str,
    module_id: impl Fn(DenseModuleId) -> String,
) -> Result<(), String> {
    let lines = code
        .lines()
        .map(crate::source_map::utf16_len)
        .collect::<Vec<_>>();
    for mapping in mappings {
        for token in &mapping.tokens {
            let module = module_id(mapping.dense_index);
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

/// Serializes readable chunk mappings after the caller supplies stable source
/// labels. Every line without an honest module token receives an explicit
/// unmapped marker so it cannot inherit the previous line's origin.
pub fn serialize_readable_source_map(
    modules: &impl ModuleMapLookup,
    labels: &HashMap<DenseModuleId, String>,
    mappings: &[ModuleMapping],
    output_name: &str,
    code: &str,
) -> String {
    let mut ordered = mappings
        .iter()
        .flat_map(|mapping| {
            mapping
                .tokens
                .iter()
                .map(move |token| (token, mapping.dense_index))
        })
        .collect::<Vec<_>>();
    ordered.sort_by_key(|(token, _)| (token.generated_line, token.generated_column));
    let mut builder = SourceMapBuilder::default();
    builder.set_file(output_name);
    let mut source_ids = HashMap::new();
    let mut index = 0;
    for line in 0..line_count(code) {
        let start = index;
        while index < ordered.len() && ordered[index].0.generated_line == line {
            index += 1;
        }
        let mut mapped = false;
        for (token, dense) in &ordered[start..index] {
            let Some((map, module_source)) = modules.module_map(*dense) else {
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
                        .add_source_and_content(label.as_str(), map.source_text(module_source));
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
            builder.add_token(line, 0, 0, 0, None, None);
        }
    }
    builder.into_sourcemap().to_json_string()
}

pub fn serialize_composed_source_map<'a>(
    modules: &impl ModuleMapLookup,
    labels: &HashMap<DenseModuleId, String>,
    readable_mappings: &[ModuleMapping],
    minified_tokens: &[oxc_sourcemap::Token],
    resolved: impl IntoIterator<Item = Option<ResolvedMinifiedToken<'a>>>,
    output_name: &str,
    chunk_name: &str,
) -> Result<String, String> {
    let mut builder = SourceMapBuilder::default();
    builder.set_file(output_name);
    let mut source_ids = HashMap::new();
    let mut mapped_any = false;
    for (minified, resolved) in minified_tokens.iter().zip(resolved) {
        let Some(token) = resolved else {
            builder.add_token(
                minified.get_dst_line(),
                minified.get_dst_col(),
                0,
                0,
                None,
                None,
            );
            continue;
        };
        let source_id = match source_ids.get(&token.dense) {
            Some(id) => *id,
            None => {
                let (Some((map, module_source)), Some(label)) =
                    (modules.module_map(token.dense), labels.get(&token.dense))
                else {
                    return Err(format!(
                        "source-map composition for chunk `{chunk_name}` resolved a token \
                         into module {}, which has no map or label",
                        token.dense
                    ));
                };
                let id =
                    builder.add_source_and_content(label.as_str(), map.source_text(module_source));
                source_ids.insert(token.dense, id);
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
    if !mapped_any
        && readable_mappings
            .iter()
            .any(|mapping| !mapping.tokens.is_empty())
    {
        return Err(format!(
            "source-map composition produced no honest mapping for minified chunk \
             `{chunk_name}`: its modules carry real printer positions, but the \
             minified->readable map resolved into none of them"
        ));
    }
    Ok(builder.into_sourcemap().to_json_string())
}

#[derive(Clone, Copy)]
pub struct FlatRenderModule<'a> {
    pub flat: &'a FlatModule,
    pub dependencies: &'a [(String, DenseModuleId, DependencyDemand)],
    pub pruned_imports: &'a HashSet<String>,
    pub map: Option<&'a ModuleSourceMap>,
    pub has_externals: bool,
    pub uses_cjs_globals: bool,
}

struct FlatStaticView<'a> {
    ids: &'a [Arc<str>],
    modules: &'a [Option<FlatRenderModule<'a>>],
}

impl StaticGraphView for FlatStaticView<'_> {
    type Dependencies<'a>
        = std::iter::FilterMap<
        std::slice::Iter<'a, (String, DenseModuleId, DependencyDemand)>,
        fn(&'a (String, DenseModuleId, DependencyDemand)) -> Option<DenseModuleId>,
    >
    where
        Self: 'a;

    fn module_id(&self, module: DenseModuleId) -> &str {
        &self.ids[module]
    }
    fn present(&self, module: DenseModuleId) -> bool {
        self.modules.get(module).is_some_and(Option::is_some)
    }
    fn dependencies(&self, module: DenseModuleId) -> Self::Dependencies<'_> {
        fn target(edge: &(String, DenseModuleId, DependencyDemand)) -> Option<DenseModuleId> {
            (!edge.2.deferred()).then_some(edge.1)
        }
        self.modules[module]
            .as_ref()
            .expect("present flat module")
            .dependencies
            .iter()
            .filter_map(target)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn render_flat(
    module_ids: &[Arc<str>],
    modules: &[Option<FlatRenderModule<'_>>],
    graph_entry: DenseModuleId,
    reachable: &[DenseModuleId],
    roots: &[DenseModuleId],
    chunk_names: &HashMap<DenseModuleId, String>,
    global_demands: &[Demand],
    is_main: bool,
    format: ModuleFormat,
    main_prelude: Option<&str>,
) -> Option<RenderedBundle> {
    let entry = match (is_main, roots) {
        (true, _) => graph_entry,
        (false, [root]) => *root,
        (false, _) => return None,
    };
    let reachable_set = reachable.iter().copied().collect::<HashSet<_>>();
    for &module in reachable {
        let module = modules.get(module)?.as_ref()?;
        if module.has_externals || (format.is_esm() && module.uses_cjs_globals) {
            return None;
        }
    }
    let mut included = reachable
        .iter()
        .copied()
        .filter(|module| modules[*module].is_some_and(|module| module.flat.has_direct_effects))
        .collect::<HashSet<_>>();
    if !is_main {
        included.insert(entry);
    }
    let mut pending = included.iter().copied().collect::<Vec<_>>();
    while let Some(source) = pending.pop() {
        for (_, target, demand) in modules[source].as_ref()?.dependencies {
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
    let view = FlatStaticView {
        ids: module_ids,
        modules,
    };
    let order = static_execution_order(&view, entry, &included)?;
    let mut declarations = HashSet::new();
    for &module in &order {
        if modules[module]
            .as_ref()?
            .flat
            .declarations
            .iter()
            .any(|name| !declarations.insert(name.clone()))
        {
            return None;
        }
    }
    let mut demands = global_demands.to_vec();
    if is_main {
        demands[entry] = Demand::default();
    } else {
        demands[entry].all = true;
    }
    let shaken = order
        .par_iter()
        .map(|&dense| -> Option<(String, Option<LineTrack>)> {
            let module = modules[dense].as_ref()?;
            let wanted = module.map.is_some() && module.flat.map_lines.is_some();
            let (mut code, shake_lines) = shake(
                &module.flat.code,
                &demands[dense],
                module.pruned_imports,
                wanted,
            );
            let mut track = match (shake_lines, module.flat.map_lines.as_ref()) {
                (Some(shake), Some(flat)) => Some(shake.compose(flat)),
                _ => None,
            };
            for (specifier, target, demand) in module.dependencies {
                if !demand.dynamic {
                    continue;
                }
                let chunk = chunk_names.get(target)?;
                let import = format!("import({})", quote(specifier));
                let lowered = format!("__dynamic(require, {})", quote(specifier));
                let replacement = if format.is_esm() {
                    format!("import({})", quote(chunk))
                } else {
                    format!("Promise.resolve().then(()=>require({}))", quote(chunk))
                };
                let needle = if code.contains(&import) {
                    import
                } else if code.contains(&lowered) {
                    lowered
                } else {
                    return None;
                };
                code = match track.as_mut() {
                    Some(track) => {
                        let mut edits = LineTrack::identity(code.lines().count());
                        let rewritten = replace_tracked(&code, &needle, &replacement, &mut edits)
                            .unwrap_or_else(|| code.clone());
                        *track = edits.compose(track);
                        rewritten
                    }
                    None => code.replace(&needle, &replacement),
                };
            }
            Some((code, track))
        })
        .collect::<Vec<_>>();
    let mut code = String::new();
    let mut mappings = Vec::with_capacity(order.len());
    let mut generated_line = 0;
    for (&dense, shaken) in order.iter().zip(&shaken) {
        let (module_code, track) = shaken.as_ref()?;
        if module_code.is_empty() {
            continue;
        }
        let mut tokens = Vec::new();
        if let (Some(track), Some(map)) = (track.as_ref(), modules[dense].as_ref()?.map) {
            track.project(map, generated_line, &mut tokens);
        }
        mappings.push(ModuleMapping {
            dense_index: dense,
            generated_line,
            tokens,
        });
        generated_line += module_code.lines().count() as u32;
        code.push_str(module_code);
    }
    if !is_main {
        let exports = modules[entry]
            .as_ref()?
            .flat
            .exports
            .iter()
            .filter(|name| demands[entry].includes(name))
            .cloned()
            .collect::<Vec<_>>();
        if format.is_esm() {
            code.push_str(&format!("export{{{}}};\n", exports.join(",")));
        } else {
            code.push_str(&format!("module.exports={{{}}};\n", exports.join(",")));
        }
    }
    if is_main && let Some(prelude) = main_prelude {
        code.insert_str(0, prelude);
        let lines = prelude.bytes().filter(|byte| *byte == b'\n').count() as u32;
        for mapping in &mut mappings {
            mapping.generated_line += lines;
            for token in &mut mapping.tokens {
                token.generated_line += lines;
            }
        }
    }
    Some(RenderedBundle {
        code,
        mappings,
        map_json: None,
    })
}

fn quote(value: &str) -> String {
    serde_json::to_string(value).expect("a string always serializes")
}

#[derive(Default)]
pub struct RenderCache {
    pub entries: HashMap<u64, RenderedBundle>,
}

pub struct RenderKeyDependency<'a> {
    pub specifier: &'a str,
    pub target: DenseModuleId,
    pub dynamic: bool,
    pub eager: bool,
    pub all: bool,
    pub names: &'a [String],
}

/// Borrowed graph view used by render-cache identity. GAT iterators let the
/// planner hash root-owned records without cloning dependency or export sets.
pub trait RenderKeyGraph {
    type ExportNames<'a>: Iterator<Item = &'a str>
    where
        Self: 'a;
    type Dependencies<'a>: Iterator<Item = RenderKeyDependency<'a>>
    where
        Self: 'a;

    fn code_hash(&self, module: DenseModuleId) -> Option<u64>;
    fn runtime_id(&self, module: DenseModuleId) -> Option<usize>;
    fn export_all(&self, module: DenseModuleId) -> bool;
    fn export_names(&self, module: DenseModuleId) -> Self::ExportNames<'_>;
    fn is_async(&self, module: DenseModuleId) -> bool;
    fn dependencies(&self, module: DenseModuleId) -> Self::Dependencies<'_>;
}

pub struct RenderKeyOptions<'a> {
    pub format: u8,
    pub any_async: bool,
    pub hmr: bool,
    pub minify: bool,
    pub source_map: bool,
    pub is_main: bool,
    pub roots: &'a [DenseModuleId],
    pub prerequisites: &'a [String],
    pub flat_allowed: bool,
}

pub fn render_key(
    graph: &impl RenderKeyGraph,
    modules: &[DenseModuleId],
    chunk_names: &HashMap<DenseModuleId, String>,
    options: RenderKeyOptions<'_>,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    options.format.hash(&mut hasher);
    options.any_async.hash(&mut hasher);
    options.hmr.hash(&mut hasher);
    options.minify.hash(&mut hasher);
    options.source_map.hash(&mut hasher);
    options.is_main.hash(&mut hasher);
    options.roots.hash(&mut hasher);
    options.prerequisites.hash(&mut hasher);
    options.flat_allowed.hash(&mut hasher);
    modules.len().hash(&mut hasher);
    for &module in modules {
        module.hash(&mut hasher);
        let Some(code_hash) = graph.code_hash(module) else {
            u64::MAX.hash(&mut hasher);
            continue;
        };
        code_hash.hash(&mut hasher);
        hash_optional_id(&mut hasher, graph.runtime_id(module));
        graph.export_all(module).hash(&mut hasher);
        let mut names = graph.export_names(module).collect::<Vec<_>>();
        names.sort_unstable();
        names.hash(&mut hasher);
        graph.is_async(module).hash(&mut hasher);
        for dependency in graph.dependencies(module) {
            dependency.specifier.hash(&mut hasher);
            dependency.dynamic.hash(&mut hasher);
            dependency.eager.hash(&mut hasher);
            dependency.all.hash(&mut hasher);
            dependency.names.hash(&mut hasher);
            dependency.target.hash(&mut hasher);
            hash_optional_id(&mut hasher, graph.runtime_id(dependency.target));
            graph.is_async(dependency.target).hash(&mut hasher);
            if dependency.dynamic {
                chunk_names.get(&dependency.target).hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}

fn hash_optional_id(hasher: &mut DefaultHasher, id: Option<usize>) {
    match id {
        Some(value) => {
            1u8.hash(hasher);
            value.hash(hasher);
        }
        None => 0u8.hash(hasher),
    }
}

/// Complete framework-neutral input to deterministic chunk partitioning.
pub struct ChunkGraph<'a> {
    pub entry: DenseModuleId,
    pub module_ids: &'a [Arc<str>],
    pub allowed: &'a HashSet<DenseModuleId>,
    pub static_edges: &'a [Vec<DenseModuleId>],
    pub dynamic_edges: &'a [Vec<DenseModuleId>],
    /// Optional integration-selected name for the private chunk owning a root.
    pub private_chunk_names: &'a HashMap<DenseModuleId, String>,
}

impl ChunkGraph<'_> {
    pub fn plan(&self, entry_file: &str) -> Result<Vec<ChunkPlan>, String> {
        let (stem, extension) = split_file_name(entry_file)?;
        let main = self.static_closure(self.entry);
        let mut roots = self
            .allowed
            .iter()
            .flat_map(|source| self.dynamic_edges[*source].iter().copied())
            .filter(|target| !main.contains(target))
            .collect::<Vec<_>>();
        roots.sort_by(|left, right| self.module_ids[*left].cmp(&self.module_ids[*right]));
        roots.dedup();

        let mut closures = Vec::with_capacity(roots.len());
        for (index, &root) in roots.iter().enumerate() {
            if !self.allowed.contains(&root) {
                return Err(format!(
                    "dynamic-import root {} (chunk {}) was dropped from the live module set; its chunk would be empty and importing it would fail at runtime",
                    self.module_ids[root],
                    index + 1
                ));
            }
            closures.push(self.static_closure(root));
        }

        let mut groups: BTreeMap<Vec<usize>, Vec<DenseModuleId>> = BTreeMap::new();
        let mut ordered = self.allowed.iter().copied().collect::<Vec<_>>();
        ordered.sort_by(|left, right| self.module_ids[*left].cmp(&self.module_ids[*right]));
        for module in ordered {
            if main.contains(&module) {
                continue;
            }
            let label = closures
                .iter()
                .enumerate()
                .filter_map(|(index, closure)| closure.contains(&module).then_some(index))
                .collect::<Vec<_>>();
            if label.is_empty() {
                return Err(format!(
                    "live module {} is in neither the entry closure nor any dynamic-import closure, so no chunk would carry it",
                    self.module_ids[module]
                ));
            }
            groups.entry(label).or_default().push(module);
        }

        let mut plans = Vec::with_capacity(groups.len());
        let mut shared_count = 0;
        for (label, members) in groups {
            let member_set = members.iter().copied().collect::<HashSet<_>>();
            let chunk_roots = label
                .iter()
                .map(|index| roots[*index])
                .filter(|root| member_set.contains(root))
                .collect::<Vec<_>>();
            let file_name = match (label.as_slice(), chunk_roots.as_slice()) {
                ([index], [root]) => self
                    .private_chunk_names
                    .get(root)
                    .map(|name| name.replace("{ext}", &extension))
                    .unwrap_or_else(|| format!("{stem}.chunk-{}{extension}", index + 1)),
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

        let mut owner = vec![None; self.module_ids.len()];
        for (index, plan) in plans.iter().enumerate() {
            for &member in &plan.members {
                owner[member] = Some(index);
            }
        }
        for (index, plan) in plans.iter_mut().enumerate() {
            let mut prerequisites = plan
                .members
                .iter()
                .flat_map(|member| self.static_edges[*member].iter())
                .filter(|target| self.allowed.contains(target))
                .filter_map(|target| owner[*target])
                .filter(|other| *other != index)
                .collect::<Vec<_>>();
            prerequisites.sort_unstable();
            prerequisites.dedup();
            plan.prerequisites = prerequisites;
        }
        Ok(plans)
    }

    fn static_closure(&self, root: DenseModuleId) -> HashSet<DenseModuleId> {
        let mut closure = HashSet::new();
        let mut queue = VecDeque::from([root]);
        while let Some(module) = queue.pop_front() {
            if !self.allowed.contains(&module) || !closure.insert(module) {
                continue;
            }
            queue.extend(self.static_edges[module].iter().copied());
        }
        closure
    }
}

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

pub fn chunk_names(plans: &[ChunkPlan]) -> HashMap<DenseModuleId, String> {
    plans
        .iter()
        .flat_map(|plan| {
            plan.roots
                .iter()
                .map(|root| (*root, format!("./{}", plan.file_name)))
        })
        .collect()
}

pub fn chunk_load_order(plans: &[ChunkPlan], index: usize) -> Vec<String> {
    fn visit(
        plans: &[ChunkPlan],
        index: usize,
        seen: &mut HashSet<usize>,
        ordered: &mut Vec<String>,
    ) {
        if !seen.insert(index) {
            return;
        }
        for &prerequisite in &plans[index].prerequisites {
            visit(plans, prerequisite, seen, ordered);
        }
        ordered.push(plans[index].file_name.clone());
    }
    let mut ordered = Vec::new();
    visit(plans, index, &mut HashSet::new(), &mut ordered);
    ordered
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_modules_form_one_disjoint_prerequisite_chunk() {
        // entry dynamically imports a and b; both statically depend on shared.
        let ids = ["entry", "a", "b", "shared"].map(Arc::<str>::from).to_vec();
        let allowed = HashSet::from([0, 1, 2, 3]);
        let static_edges = vec![vec![], vec![3], vec![3], vec![]];
        let dynamic_edges = vec![vec![1, 2], vec![], vec![], vec![]];
        let plans = ChunkGraph {
            entry: 0,
            module_ids: &ids,
            allowed: &allowed,
            static_edges: &static_edges,
            dynamic_edges: &dynamic_edges,
            private_chunk_names: &HashMap::new(),
        }
        .plan("client.js")
        .unwrap();

        assert_eq!(plans.len(), 3);
        let shared = plans.iter().position(|plan| plan.members == [3]).unwrap();
        assert!(plans[shared].file_name.starts_with("client.shared-"));
        for root in [1, 2] {
            let plan = plans.iter().find(|plan| plan.roots == [root]).unwrap();
            assert_eq!(plan.prerequisites, [shared]);
        }
        let members = plans
            .iter()
            .flat_map(|plan| plan.members.iter().copied())
            .collect::<HashSet<_>>();
        assert_eq!(members, HashSet::from([1, 2, 3]));
    }

    #[test]
    fn an_integration_can_name_one_private_root_chunk() {
        let ids = vec![Arc::from("entry"), Arc::from("virtual")];
        let allowed = HashSet::from([0, 1]);
        let static_edges = vec![vec![], vec![]];
        let dynamic_edges = vec![vec![1], vec![]];
        let names = HashMap::from([(1, "integration-manifest.js".into())]);
        let plans = ChunkGraph {
            entry: 0,
            module_ids: &ids,
            allowed: &allowed,
            static_edges: &static_edges,
            dynamic_edges: &dynamic_edges,
            private_chunk_names: &names,
        }
        .plan("client.js")
        .unwrap();
        assert_eq!(plans[0].file_name, "integration-manifest.js");
    }

    struct KeyGraph {
        hashes: Vec<u64>,
        names: Vec<Vec<&'static str>>,
    }

    impl RenderKeyGraph for KeyGraph {
        type ExportNames<'a> =
            std::iter::Map<std::slice::Iter<'a, &'static str>, fn(&'a &'static str) -> &'a str>;
        type Dependencies<'a> = std::iter::Empty<RenderKeyDependency<'a>>;

        fn code_hash(&self, module: usize) -> Option<u64> {
            self.hashes.get(module).copied()
        }
        fn runtime_id(&self, module: usize) -> Option<usize> {
            Some(module)
        }
        fn export_all(&self, _module: usize) -> bool {
            false
        }
        fn export_names(&self, module: usize) -> Self::ExportNames<'_> {
            fn borrow<'a>(value: &'a &'static str) -> &'a str {
                value
            }
            self.names[module].iter().map(borrow)
        }
        fn is_async(&self, _module: usize) -> bool {
            false
        }
        fn dependencies(&self, _module: usize) -> Self::Dependencies<'_> {
            std::iter::empty()
        }
    }

    fn key(graph: &KeyGraph, modules: &[usize], minify: bool) -> u64 {
        render_key(
            graph,
            modules,
            &HashMap::new(),
            RenderKeyOptions {
                format: 0,
                any_async: false,
                hmr: false,
                minify,
                source_map: false,
                is_main: true,
                roots: &[0],
                prerequisites: &[],
                flat_allowed: true,
            },
        )
    }

    #[test]
    fn render_keys_are_option_sensitive_but_chunk_local() {
        let graph = KeyGraph {
            hashes: vec![10, 20],
            names: vec![vec!["default"], vec![]],
        };
        let baseline = key(&graph, &[0], false);
        assert_ne!(baseline, key(&graph, &[0], true));

        let unrelated_changed = KeyGraph {
            hashes: vec![10, 999],
            names: graph.names.clone(),
        };
        assert_eq!(baseline, key(&unrelated_changed, &[0], false));
        assert_ne!(baseline, key(&unrelated_changed, &[0, 1], false));
    }

    #[test]
    fn mapping_validation_rejects_positions_outside_the_chunk() {
        let mappings = vec![ModuleMapping {
            dense_index: 0,
            generated_line: 0,
            tokens: vec![MapToken {
                generated_line: 1,
                generated_column: 0,
                source_line: 0,
                source_column: 0,
                name: None,
            }],
        }];
        let error = validate_mappings("one line\n", &mappings, "client.js", |_| {
            "entry.js".to_string()
        })
        .unwrap_err();
        assert!(error.contains("entry.js"));
        assert!(error.contains("only 1 lines"));
    }

    #[test]
    fn flat_render_orders_dependencies_and_exports_a_dynamic_root() {
        let ids = vec![Arc::from("entry"), Arc::from("lazy")];
        let entry_flat = FlatModule {
            code: "console.log('entry');\n".into(),
            map_lines: None,
            declarations: vec![],
            exports: vec![],
            has_direct_effects: true,
            import_replacements: vec![],
            foldable: None,
        };
        let lazy_flat = FlatModule {
            code: "const value=1;\n".into(),
            map_lines: None,
            declarations: vec!["value".into()],
            exports: vec!["value".into()],
            has_direct_effects: false,
            import_replacements: vec![],
            foldable: None,
        };
        let entry_dependencies = vec![];
        let lazy_dependencies = vec![];
        let pruned = HashSet::new();
        let modules = vec![
            Some(FlatRenderModule {
                flat: &entry_flat,
                dependencies: &entry_dependencies,
                pruned_imports: &pruned,
                map: None,
                has_externals: false,
                uses_cjs_globals: false,
            }),
            Some(FlatRenderModule {
                flat: &lazy_flat,
                dependencies: &lazy_dependencies,
                pruned_imports: &pruned,
                map: None,
                has_externals: false,
                uses_cjs_globals: false,
            }),
        ];
        let mut demands = vec![Demand::default(), Demand::default()];
        demands[1].all = true;
        let rendered = render_flat(
            &ids,
            &modules,
            0,
            &[1],
            &[1],
            &HashMap::new(),
            &demands,
            false,
            ModuleFormat::Esm,
            None,
        )
        .unwrap();
        assert!(rendered.code.contains("const value=1"));
        assert!(rendered.code.ends_with("export{value};\n"));
    }

    #[test]
    fn runtime_fragments_own_factory_and_chunk_map_generation() {
        let dependency = DependencyDemand {
            specifier: "./lazy".into(),
            dynamic: true,
            ..DependencyDemand::default()
        };
        let dependencies = vec![("./lazy".to_string(), 1, dependency)];
        let no_dependencies = vec![];
        let pruned = HashSet::new();
        let modules = vec![
            Some(RuntimeRenderModule {
                id: "/entry.js",
                code: "const lazy=require.dynamic(\"./lazy\");",
                dependencies: &dependencies,
                pruned_imports: &pruned,
                map: None,
                uses_dirname: true,
            }),
            Some(RuntimeRenderModule {
                id: "/lazy.js",
                code: "exports.value=1;",
                dependencies: &no_dependencies,
                pruned_imports: &pruned,
                map: None,
                uses_dirname: false,
            }),
        ];
        let mut chunk_names = HashMap::new();
        chunk_names.insert(1, "./lazy.js".to_string());
        let fragments = render_runtime_fragments(
            &modules,
            &[0, 1],
            &[0],
            &chunk_names,
            &[Some(10), Some(11)],
            &[Demand::default(), Demand::default()],
            &AsyncModules::default(),
            ModuleFormat::BrowserEsm,
            &|| false,
        )
        .unwrap();
        assert_eq!(fragments.len(), 2);
        assert!(fragments[0].module.starts_with("10:function("));
        assert!(
            fragments[0]
                .module
                .contains("const __filename=\"/index.js\"")
        );
        assert!(fragments[0].import_map.contains("\"./lazy\":11"));
        assert!(
            fragments[0]
                .chunk_map
                .contains("\"./lazy\":[\"./lazy.js\",11]")
        );
    }

    #[test]
    fn runtime_header_orders_host_and_compatibility_preludes() {
        let header = render_runtime_header(
            ModuleFormat::Esm,
            &["./shared.js".to_string()],
            &["host();\n", "compat();\n"],
        );
        assert!(header.prelude.starts_with("host();"));
        assert!(
            header.prelude.find("host();").unwrap() < header.prelude.find("compat();").unwrap()
        );
        assert_eq!(header.prerequisite_loads, "import \"./shared.js\";\n");
        assert_eq!(
            header.generated_lines as usize,
            header.prelude.lines().count() + 1
        );
    }

    #[test]
    fn runtime_literal_assembly_tracks_factory_regions() {
        let literals = assemble_runtime_literals(
            vec![RuntimeFragment {
                dense_index: 2,
                module: "7:function(){\nbody();\n},\n".to_string(),
                import_map: "7:{},\n".to_string(),
                chunk_map: "7:{},\n".to_string(),
                generated_lines: 1,
                track: None,
            }],
            4,
            |_| None,
        );
        assert!(literals.modules.contains("body()"));
        assert_eq!(literals.import_maps, "7:{},\n");
        assert_eq!(literals.chunk_maps, "7:{},\n");
        assert_eq!(literals.mappings[0].generated_line, 7);
    }
}
