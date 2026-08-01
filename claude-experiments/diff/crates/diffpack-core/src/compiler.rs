use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use oxc_allocator::Allocator;
use oxc_ast::{
    ast::{
        BindingPattern, Declaration, ExportDefaultDeclarationKind, Expression,
        ImportDeclarationSpecifier, Program, Statement, VariableDeclarationKind,
    },
    builder::{AstBuilder, NONE},
};
use oxc_ast_visit::{Visit, VisitMut, walk, walk_mut};
use oxc_codegen::{Codegen, CodegenOptions, Context, Gen};
use oxc_ecmascript::BoundNames;
use oxc_parser::Parser;
use oxc_semantic::{Scoping, SemanticBuilder};
use oxc_span::SourceType;
use oxc_syntax::{operator::BinaryOperator, symbol::SymbolId};
use oxc_transformer::{ReactRefreshOptions, TransformOptions, Transformer};

use crate::frontend_profile::{self, Phase};
use crate::parser::{JsxExtensions, collect_dependencies, collect_dynamic_dependencies};
use crate::source_map::{LineTrack, MapOrigin, MapToken, ModuleSourceMap};
pub use crate::transform::{
    BodyUse, DecoratorConfig, DependencyDemand, FlatModule, FoldExpression, FoldableModule,
    JsxConfig, JsxRuntime, ModuleLiveness, ProjectConfig, ReExport, SourceLanguage, Target,
    TransformDiagnostic, TransformResult,
};

/// Description of source prepared by framework/default-loader hooks before the
/// framework-independent compiler parses it.
#[derive(Debug, Clone, Copy)]
pub struct PreparedSource<'a> {
    pub code: &'a str,
    pub force_jsx: bool,
    pub map_origin: MapOrigin,
}

/// Inputs a host-owned source policy supplies to the generic compiler pipeline.
/// Framework integrations can prepare or rewrite source while the graph driver
/// remains unaware of which integration produced the result.
pub struct CompileRequest<'a> {
    pub path: &'a Path,
    pub source: &'a str,
    pub target: Target,
    pub hmr: bool,
    pub refresh: bool,
    pub jsx: JsxExtensions,
    pub project_config: &'a ProjectConfig,
    pub language: SourceLanguage,
    pub source_maps: bool,
}

/// Host policy used by a graph driver to compile one source module.
pub trait ModuleCompiler: Send + Sync {
    fn compile(&self, request: CompileRequest<'_>) -> TransformResult;

    /// Whether loader-side project configuration should treat this path as a
    /// generated adapter module rather than application source.
    fn is_generated_path(&self, _path: &Path) -> bool {
        false
    }

    /// Optional integration-specific advice appended to an unresolved import.
    fn unresolved_import_help(&self, _specifier: &str) -> Option<&'static str> {
        None
    }
}

/// The framework-independent JavaScript/TypeScript compiler policy.
#[derive(Debug, Default, Clone, Copy)]
pub struct CoreModuleCompiler;

impl ModuleCompiler for CoreModuleCompiler {
    fn compile(&self, request: CompileRequest<'_>) -> TransformResult {
        transform_module_in_language(
            request.path,
            request.source,
            request.target,
            request.refresh,
            request.jsx,
            request.project_config,
            request.language,
            request.source_maps,
        )
    }
}

/// Optional semantic AST specialization supplied by an integration crate after
/// Oxc lowering and before core computes dependency demand and liveness.
pub trait SemanticTransform {
    fn apply<'a>(
        &self,
        allocator: &'a Allocator,
        program: &mut Program<'a>,
        scoping: &Scoping,
        target: Target,
        path: &Path,
    ) -> bool;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NoSemanticTransform;

impl SemanticTransform for NoSemanticTransform {
    fn apply<'a>(
        &self,
        _allocator: &'a Allocator,
        _program: &mut Program<'a>,
        _scoping: &Scoping,
        _target: Target,
        _path: &Path,
    ) -> bool {
        false
    }
}

/// Converts an oxc diagnostic collection, preserving severity. oxc documents a
/// missing severity as an error, and `Severity::Advice` is informational, so
/// only `Severity::Error` is fatal.
fn transform_diagnostics(diagnostics: oxc_diagnostics::Diagnostics) -> Vec<TransformDiagnostic> {
    diagnostics
        .into_vec()
        .into_iter()
        .map(|diagnostic| TransformDiagnostic {
            fatal: diagnostic.severity == oxc_diagnostics::Severity::Error,
            message: diagnostic.to_string(),
        })
        .collect()
}

/// Replaces oxc's bare `Unexpected JSX expression` with an actionable message
/// when the configured parser capability does not enable JSX for this extension.
fn explain_jsx_in_non_jsx(
    path: &Path,
    source_type: SourceType,
    diagnostics: &mut [TransformDiagnostic],
) {
    if source_type.is_jsx() {
        return;
    }
    let Some(extension) = path.extension().and_then(|extension| extension.to_str()) else {
        return;
    };
    // Only the extensions this rule actually governs. A `.vue`/`.svelte` file also
    // fails with "Unexpected JSX expression", but its remedy is a component
    // compiler, not a rename — that is a separate defect and is left untouched.
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("module");
    let explanation = match extension {
        "js" | "mjs" | "cjs" => format!(
            "JSX is not enabled for `.{extension}` files by the configured parser policy. \
             Rename it to `{stem}.jsx` or configure the host integration to enable JSX in \
             JavaScript extensions."
        ),
        "ts" | "mts" | "cts" => format!(
            "JSX is not enabled for `.{extension}` files: in TypeScript `<T>x` is a type \
             assertion, so a module containing JSX must be `{stem}.tsx`."
        ),
        _ => return,
    };
    for diagnostic in diagnostics.iter_mut() {
        if diagnostic.message.contains("Unexpected JSX expression") {
            diagnostic.message = explanation.clone();
        }
    }
}

pub fn transform_module(path: &Path, source: &str, target: Target) -> TransformResult {
    transform_module_with_options(
        path,
        source,
        target,
        false,
        JsxExtensions::default(),
        &JsxConfig::default(),
    )
}

/// Transforms one module to standalone ESM using the default explicit JSX-file
/// policy. Host integrations needing JSX in JavaScript extensions use
/// [`transform_to_standalone_esm_with_jsx`].
pub fn transform_to_standalone_esm(path: &Path, source: &str) -> Result<String, String> {
    transform_to_standalone_esm_with_jsx(path, source, JsxExtensions::default())
}

/// Transforms one module to standalone ESM without rewriting its specifiers.
pub fn transform_to_standalone_esm_with_jsx(
    path: &Path,
    source: &str,
    jsx_extensions: JsxExtensions,
) -> Result<String, String> {
    let allocator = Allocator::default();
    let source_type = crate::parser::source_type_for(path, jsx_extensions);
    let parsed = Parser::new(&allocator, source, source_type).parse();
    let mut diagnostics: Vec<String> = parsed
        .diagnostics
        .into_iter()
        .map(|d| d.to_string())
        .collect();
    let mut program = parsed.program;
    let semantic = SemanticBuilder::new().build(&program);
    diagnostics.extend(semantic.diagnostics.into_iter().map(|d| d.to_string()));
    // Default TransformOptions lowers JSX with the automatic runtime (react) and strips
    // TS, emitting `import { jsx as _jsx } from "react/jsx-runtime"` — exactly what a
    // Node run of a `@vercel/og` generator needs. No refresh, no bundler rewrite.
    // Deliberately EXEMPT from the project's [`JsxConfig`]: `@vercel/og` renders with
    // a real React, and the adapter writes a `react/jsx-runtime` stand-in for this
    // prerender specifically — an app-level `jsxImportSource` would point it at a
    // package that is not there.
    let transform_options = TransformOptions::default();
    let transformed = Transformer::new(&allocator, path, &transform_options)
        .build_with_scoping(semantic.semantic.into_scoping(), &mut program);
    diagnostics.extend(transformed.diagnostics.into_iter().map(|d| d.to_string()));
    if !diagnostics.is_empty() {
        return Err(format!(
            "cannot transform {} for the @vercel/og prerender: {}",
            path.display(),
            diagnostics.join("; "),
        ));
    }
    Ok(Codegen::new().build(&program).code)
}

/// Like [`transform_module`], but with `refresh` enabling oxc's native React Fast
/// Refresh transform: it injects the per-component `$RefreshReg$` registrations and
/// `$RefreshSig$` hook signatures the React Refresh runtime needs to swap a
/// component in place while preserving state. Enabled by the dev server for client
/// component modules (never by `build-app`). This is the full transform equivalent
/// to `react-refresh/babel` — done natively in oxc, no Node — replacing the earlier
/// footer-only registration that only worked for simple/default-export components.
///
/// `jsx` is the host's JSX-extension capability (see [`JsxExtensions`]);
/// `jsx_config` controls how this module's JSX is lowered. Both are resolved by
/// the caller before entering core.
pub fn transform_module_with_options(
    path: &Path,
    source: &str,
    target: Target,
    refresh: bool,
    jsx: JsxExtensions,
    jsx_config: &JsxConfig,
) -> TransformResult {
    transform_module_in_language(
        path,
        source,
        target,
        refresh,
        jsx,
        &ProjectConfig {
            jsx: jsx_config.clone(),
            decorators: DecoratorConfig::default(),
        },
        SourceLanguage::FromPath,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn transform_module_in_language(
    path: &Path,
    source: &str,
    target: Target,
    refresh: bool,
    jsx: JsxExtensions,
    project_config: &ProjectConfig,
    language: SourceLanguage,
    source_maps: bool,
) -> TransformResult {
    transform_prepared_module_in_language(
        path,
        PreparedSource {
            code: source,
            force_jsx: false,
            map_origin: MapOrigin::File,
        },
        target,
        refresh,
        jsx,
        project_config,
        language,
        source_maps,
    )
}

/// Which language `source` is written in, when the module's own path cannot say.
///
/// A path answers this for every file diffpack reads off disk, but not for source
/// a compiler produced: `App.vue` compiled by `@vue/compiler-sfc` is TypeScript
/// whenever the SFC's `<script>` was, and `.vue` names neither. This is the
/// caller's explicit answer for those, and is exactly the choice
/// an external component compiler makes when handing TypeScript output to core.
/// [`transform_module_with_options`] for source a component compiler produced:
/// the path is the component (`App.vue`), but the language is `language`, not
/// whatever the extension implies. See [`crate::sfc`].
#[allow(clippy::too_many_arguments)]
pub fn transform_prepared_module_in_language(
    path: &Path,
    prepared: PreparedSource<'_>,
    target: Target,
    refresh: bool,
    jsx: JsxExtensions,
    project_config: &ProjectConfig,
    language: SourceLanguage,
    source_maps: bool,
) -> TransformResult {
    transform_prepared_module_in_language_with(
        path,
        prepared,
        target,
        refresh,
        jsx,
        project_config,
        language,
        source_maps,
        &NoSemanticTransform,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn transform_prepared_module_in_language_with<H: SemanticTransform>(
    path: &Path,
    prepared: PreparedSource<'_>,
    target: Target,
    refresh: bool,
    jsx: JsxExtensions,
    project_config: &ProjectConfig,
    language: SourceLanguage,
    source_maps: bool,
    semantic_transform: &H,
) -> TransformResult {
    let source = prepared.code;
    let force_jsx = prepared.force_jsx;
    let map_origin = prepared.map_origin;
    if path
        .extension()
        .is_some_and(|extension| extension == "json")
    {
        return TransformResult {
            code: format!("module.exports = {source};\n"),
            diagnostics: Vec::new(),
            is_esm: false,
            dependencies: Vec::new(),
            dependency_demands: Vec::new(),
            flat_module: None,
            liveness: ModuleLiveness::default(),
            uses_top_level_await: false,
            uses_import_meta: false,
            uses_cjs_globals: false,
            uses_dirname: false,
            workers: Vec::new(),
            map: None,
        };
    }

    let transform_started = frontend_profile::start();
    let allocator = Allocator::default();
    let source_type = if force_jsx {
        SourceType::default()
            .with_typescript(true)
            .with_jsx(true)
            .with_module(true)
    } else {
        language.source_type(path, jsx)
    };
    let parsed = Parser::new(&allocator, source, source_type).parse();
    let mut diagnostics = transform_diagnostics(parsed.diagnostics);
    explain_jsx_in_non_jsx(path, source_type, &mut diagnostics);
    let mut program = parsed.program;

    let semantic = SemanticBuilder::new()
        .with_excess_capacity(2.0)
        .with_enum_eval(true)
        .build(&program);
    diagnostics.extend(transform_diagnostics(semantic.diagnostics));
    let mut transform_options = TransformOptions::default();
    // Where the runtime helpers a lowering CALLS come from. oxc imports them rather
    // than inlining (`__decorate` for a legacy decorator, today's only case), and its
    // default names `@oxc-project/runtime`, which is not a required host dependency,
    // so a decorator would lower into an unresolvable import. The host serves them
    // from its own binary instead; see [`crate::runtime_helpers`].
    transform_options.helper_loader.module_name = std::borrow::Cow::Borrowed("@diffpack/runtime");
    // How this file's JSX is lowered. A file-level `@jsxImportSource`/`@jsx`/
    // `@jsxFrag`/`@jsxRuntime` pragma still wins: oxc rescans the program's leading
    // comments and overrides these options before building the JSX pass.
    project_config.jsx.apply(&mut transform_options);
    // How this file's `@decorator`s are lowered, from the tsconfig that owns it.
    project_config.decorators.apply(&mut transform_options);
    if !project_config.decorators.legacy
        && let Some(name) = first_decorator_name(&program)
    {
        // Stage 3 decorators. oxc lowers only TypeScript's legacy ones, so leaving
        // this alone would emit `@dec` into the bundle — syntax no engine parses,
        // failing at load with an opaque `SyntaxError: Invalid or unexpected token`
        // pointing at a minified line. Refuse HERE, naming the file and the decorator.
        diagnostics.push(TransformDiagnostic::error(format!(
            "{}: `@{name}` is a Stage 3 (TC39) decorator, which this build cannot lower — \
             the emitted bundle would carry `@{name}` verbatim and no JavaScript engine \
             parses that. Only TypeScript's legacy decorators are lowered: set \
             \"experimentalDecorators\": true in the tsconfig.json/jsconfig.json that owns \
             this file (which is also what makes a decorator receive TypeScript's \
             `(target, key, descriptor)` arguments rather than the Stage 3 \
             `(value, context)` pair, so the two are not interchangeable).",
            path.display(),
        )));
    }
    if refresh {
        // Native React Fast Refresh: oxc injects `$RefreshSig$()`/`$RefreshReg$`
        // per component (the `react-refresh/babel` equivalent), so a hot update
        // swaps the component type while preserving hook state. The globals it
        // references are installed by the dev client preamble.
        transform_options.jsx.refresh = Some(ReactRefreshOptions::default());
    }
    let transformed = Transformer::new(&allocator, path, &transform_options)
        .build_with_scoping(semantic.semantic.into_scoping(), &mut program);
    diagnostics.extend(transform_diagnostics(transformed.diagnostics));

    frontend_profile::finish(Phase::Transform, transform_started);

    // Let an integration specialize the lowered AST before demand is computed.
    // A hook may delete references, so rebuild scoping whenever it reports a
    // change; liveness and dependency demand then observe the specialized tree.
    let mut scoping = transformed.scoping;
    if semantic_transform.apply(&allocator, &mut program, &scoping, target, path) {
        scoping = SemanticBuilder::new()
            .with_excess_capacity(2.0)
            .with_enum_eval(true)
            .build(&program)
            .semantic
            .into_scoping();
    }

    // Capture the module's export/import structure BEFORE `lower_module_ast`
    // rewrites import references, so re-export edges and body uses are read from
    // the original ESM shape.
    let liveness = collect_liveness(&program, &scoping);

    // Detect constructs whose validity depends on the OUTPUT format, before
    // lowering rewrites the tree: a top-level `await` (only representable when
    // the module's statements stay at the top level of an ESM chunk) and any
    // remaining `import.meta` (an opted-in `import.meta.env` was already
    // rewritten from the source; whatever is left survives into the output).
    let mut format_scan = FormatSensitiveScan::default();
    format_scan.visit_program(&program);

    // `new Worker(new URL('./x', import.meta.url))` (module workers): the URL's
    // string literal is rewritten to a deterministic placeholder and the
    // specifier recorded; the bundler resolves it, bundles the worker entry as
    // its own self-contained file under `assets/`, and the emit substitutes the
    // real public URL. Left alone, the raw specifier would ship and 404 at
    // runtime — a silently broken feature.
    let mut worker_rewriter = WorkerRewriter {
        builder: AstBuilder::new(&allocator),
        importer: path,
        workers: Vec::new(),
    };
    worker_rewriter.visit_program(&mut program);
    let workers = worker_rewriter.workers;
    // A free reference to a CommonJS ambient (`exports`, `module`, ...) means the
    // module's code only makes sense inside a CJS-style wrapper. The registry
    // runtime's factories provide those; the flat ESM concatenation does not, so
    // such a module must never be scope-hoisted into ESM output.
    let uses_cjs_globals = ["exports", "module", "require", "__filename", "__dirname"]
        .iter()
        .any(|name| scoping.root_unresolved_references().contains_key(*name));
    // The two CommonJS ambients a browser target cannot supply from the host: a
    // browser bundle has no file location to derive them from, so the emit defines
    // them per module (see `render_runtime`). Tracked separately from
    // `uses_cjs_globals` so the definitions are emitted ONLY where they are read.
    let uses_dirname = ["__filename", "__dirname"]
        .iter()
        .any(|name| scoping.root_unresolved_references().contains_key(*name));

    let map_request = source_maps.then(|| MapRequest {
        path,
        origin: map_origin,
        // Only a GENERATED source needs its text carried here: for a real file
        // the bundler already holds the module's source and inlines that.
        source_text: matches!(map_origin, MapOrigin::Generated(_))
            .then(|| Arc::<str>::from(source)),
    });

    let lower_started = frontend_profile::start();
    let lowered = lower_module_ast(&allocator, &mut program, &scoping, map_request.as_ref());
    frontend_profile::finish(Phase::Lower, lower_started);
    if let Some(problem) = lowered.map_problem {
        // The module's own map could not be proved correct. The module still
        // builds — its bytes are unaffected — but nothing in it will be mapped,
        // and the reason is reported rather than silently swallowed.
        diagnostics.push(TransformDiagnostic {
            fatal: false,
            message: format!(
                "{}: no source map for this module — {problem}",
                path.display()
            ),
        });
    }
    TransformResult {
        code: lowered.code,
        diagnostics,
        is_esm: lowered.is_esm,
        dependencies: lowered.dependencies,
        dependency_demands: lowered.dependency_demands,
        flat_module: lowered.flat_module,
        liveness,
        uses_top_level_await: format_scan.top_level_await,
        uses_import_meta: format_scan.import_meta,
        uses_cjs_globals,
        uses_dirname,
        workers,
        map: lowered.map,
    }
}

/// The name of the FIRST `@decorator` anywhere in `program` (class, method,
/// property, accessor or parameter), for the refusal message. `None` when the module
/// has none, which is the overwhelmingly common case and the only thing the caller
/// needs to know to stay silent.
fn first_decorator_name(program: &Program<'_>) -> Option<String> {
    #[derive(Default)]
    struct FindDecorator {
        found: Option<String>,
    }
    impl<'a> Visit<'a> for FindDecorator {
        fn visit_decorator(&mut self, decorator: &oxc_ast::ast::Decorator<'a>) {
            if self.found.is_none() {
                self.found = Some(decorator_expression_name(&decorator.expression));
            }
        }
    }
    /// A readable source-shaped name for the decorator: `@Memoize`, `@Memoize(...)`
    /// and `@cache.Memoize` all report the identifier an author would recognize.
    fn decorator_expression_name(expression: &Expression<'_>) -> String {
        match expression {
            Expression::Identifier(identifier) => identifier.name.to_string(),
            Expression::CallExpression(call) => decorator_expression_name(&call.callee),
            Expression::StaticMemberExpression(member) => format!(
                "{}.{}",
                decorator_expression_name(&member.object),
                member.property.name,
            ),
            _ => "decorator".to_string(),
        }
    }
    let mut finder = FindDecorator::default();
    finder.visit_program(program);
    finder.found
}

/// Finds the two constructs whose validity depends on the output format: a
/// top-level `await` (including `for await` at the top level) and any
/// `import.meta` reference. Function bodies do not count for `await` (an
/// `async` function's await is fine anywhere) but DO count for `import.meta`
/// (it is a syntax error anywhere in a CommonJS file).
#[derive(Default)]
struct FormatSensitiveScan {
    function_depth: usize,
    top_level_await: bool,
    import_meta: bool,
}

impl<'a> Visit<'a> for FormatSensitiveScan {
    fn visit_function(
        &mut self,
        function: &oxc_ast::ast::Function<'a>,
        flags: oxc_syntax::scope::ScopeFlags,
    ) {
        self.function_depth += 1;
        walk::walk_function(self, function, flags);
        self.function_depth -= 1;
    }

    fn visit_arrow_function_expression(
        &mut self,
        arrow: &oxc_ast::ast::ArrowFunctionExpression<'a>,
    ) {
        self.function_depth += 1;
        walk::walk_arrow_function_expression(self, arrow);
        self.function_depth -= 1;
    }

    fn visit_await_expression(&mut self, expression: &oxc_ast::ast::AwaitExpression<'a>) {
        if self.function_depth == 0 {
            self.top_level_await = true;
        }
        walk::walk_await_expression(self, expression);
    }

    fn visit_for_of_statement(&mut self, statement: &oxc_ast::ast::ForOfStatement<'a>) {
        if statement.r#await && self.function_depth == 0 {
            self.top_level_await = true;
        }
        walk::walk_for_of_statement(self, statement);
    }

    fn visit_meta_property(&mut self, meta: &oxc_ast::ast::MetaProperty<'a>) {
        if meta.meta.name == "import" && meta.property.name == "meta" {
            self.import_meta = true;
        }
        walk::walk_meta_property(self, meta);
    }
}

/// The deterministic key for one worker creation site: importer path +
/// specifier, hashed. Both the placeholder in the code and the emitted worker
/// file name derive from it, so they agree by construction.
pub fn worker_key(importer: &Path, specifier: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    importer.hash(&mut hasher);
    specifier.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Rewrites `new Worker(new URL('<lit>', import.meta.url), ...)` (and
/// `SharedWorker`) URL literals to `__diffpack_worker__<key>__` placeholders,
/// recording `(key, specifier)`.
struct WorkerRewriter<'a, 'p> {
    builder: AstBuilder<'a>,
    importer: &'p Path,
    workers: Vec<(String, String)>,
}

impl<'a> VisitMut<'a> for WorkerRewriter<'a, '_> {
    fn visit_new_expression(&mut self, new_expression: &mut oxc_ast::ast::NewExpression<'a>) {
        let is_worker = matches!(
            &new_expression.callee,
            oxc_ast::ast::Expression::Identifier(identifier)
                if identifier.name == "Worker" || identifier.name == "SharedWorker"
        );
        if is_worker
            && let Some(oxc_ast::ast::Argument::NewExpression(url)) =
                new_expression.arguments.first_mut()
            && matches!(
                &url.callee,
                oxc_ast::ast::Expression::Identifier(identifier) if identifier.name == "URL"
            )
            && url.arguments.len() == 2
            && matches!(
                url.arguments.get(1).and_then(|argument| argument.as_expression()),
                Some(oxc_ast::ast::Expression::StaticMemberExpression(member))
                    if member.property.name == "url"
            )
            && let Some(oxc_ast::ast::Argument::StringLiteral(literal)) = url.arguments.first_mut()
        {
            let specifier = literal.value.to_string();
            let key = worker_key(self.importer, &specifier);
            literal.value = oxc_allocator::FromIn::from_in(
                format!("__diffpack_worker__{key}__"),
                self.builder.allocator,
            );
            literal.raw = None;
            self.workers.push((key, specifier));
            return;
        }
        walk_mut::walk_new_expression(self, new_expression);
    }
}

/// Collects the [`ModuleLiveness`] structure the cross-module dead-module
/// elimination pass needs: which of this module's own exports forward an
/// imported binding (a re-export, conditional on that export being used) versus
/// which imported names are referenced in real module code (a body use, applied
/// unconditionally once the module runs).
fn collect_liveness(program: &Program<'_>, scoping: &Scoping) -> ModuleLiveness {
    // Map each imported *local* binding to where it came from, so a bare
    // `export { local }` (no source) can be recognised as a re-export.
    let mut named_imports: HashMap<String, (String, String)> = HashMap::new();
    let mut namespace_imports: HashMap<String, String> = HashMap::new();
    let mut default_imports: HashMap<String, String> = HashMap::new();
    let mut import_symbols: HashMap<String, SymbolId> = HashMap::new();
    for statement in &program.body {
        let Statement::ImportDeclaration(declaration) = statement else {
            continue;
        };
        let specifier = declaration.source.value.to_string();
        let Some(specifiers) = &declaration.specifiers else {
            continue;
        };
        for import in specifiers {
            let local = import.local().name.to_string();
            import_symbols.insert(local.clone(), import.local().symbol_id());
            match import {
                ImportDeclarationSpecifier::ImportSpecifier(import) => {
                    named_imports.insert(
                        local,
                        (specifier.clone(), import.imported.name().to_string()),
                    );
                }
                ImportDeclarationSpecifier::ImportNamespaceSpecifier(_) => {
                    namespace_imports.insert(local, specifier.clone());
                }
                ImportDeclarationSpecifier::ImportDefaultSpecifier(_) => {
                    default_imports.insert(local, specifier.clone());
                }
            }
        }
    }

    // Symbols referenced in real module code — everything except the local names
    // inside a bare (no-source) `export { ... }` specifier list, which are pure
    // forwarding and must not be treated as a body use.
    let mut body = BodyUseCollector {
        scoping,
        used: std::collections::HashSet::new(),
        requires: std::collections::BTreeSet::new(),
    };
    body.visit_program(program);

    let mut liveness = ModuleLiveness::default();
    // Accumulate body-use demand per specifier.
    let mut body_all: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut body_names: std::collections::BTreeMap<String, std::collections::BTreeSet<String>> =
        std::collections::BTreeMap::new();
    // A `require("x")` is an unconditional whole-module use of `x`. Without this, a
    // module reached ONLY through `require` carried no demand at all, so a package
    // marked `"sideEffects": false` (which authorizes dropping an undemanded module)
    // was silently deleted from the graph — the `require` call survived, missed the
    // registry, and fell through to the host/browser external path. That is how
    // A destructured require inside a published configuration module became
    // a runtime `Cannot require ... in the browser` that killed hydration.
    body_all.extend(body.requires.iter().cloned());
    for (local, symbol) in &import_symbols {
        if !body.used.contains(symbol) {
            continue;
        }
        if let Some((specifier, imported)) = named_imports.get(local) {
            body_names
                .entry(specifier.clone())
                .or_default()
                .insert(imported.clone());
        } else if let Some(specifier) = namespace_imports.get(local) {
            body_all.insert(specifier.clone());
        } else if let Some(specifier) = default_imports.get(local) {
            body_names
                .entry(specifier.clone())
                .or_default()
                .insert("default".to_string());
        }
    }

    for statement in &program.body {
        match statement {
            Statement::ExportNamedDeclaration(export) => {
                if let Some(source) = &export.source {
                    // `export { imported as exported } from S` — a direct
                    // re-export edge for each specifier.
                    let specifier = source.value.to_string();
                    for export_specifier in &export.specifiers {
                        let exported = export_specifier.exported.name().to_string();
                        liveness.exports.push(exported.clone());
                        liveness.reexports.push(ReExport {
                            specifier: specifier.clone(),
                            imported: export_specifier.local.name().to_string(),
                            exported,
                        });
                    }
                } else if let Some(declaration) = &export.declaration {
                    declaration.bound_names(&mut |identifier| {
                        liveness.exports.push(identifier.name.to_string());
                    });
                } else {
                    // `export { local as exported }` (no source): a re-export when
                    // `local` is an imported binding, otherwise a local export.
                    for export_specifier in &export.specifiers {
                        let local = export_specifier.local.name().to_string();
                        let exported = export_specifier.exported.name().to_string();
                        liveness.exports.push(exported.clone());
                        if let Some((specifier, imported)) = named_imports.get(&local) {
                            liveness.reexports.push(ReExport {
                                specifier: specifier.clone(),
                                imported: imported.clone(),
                                exported,
                            });
                        } else if let Some(specifier) = namespace_imports.get(&local) {
                            liveness.reexports.push(ReExport {
                                specifier: specifier.clone(),
                                imported: "*".to_string(),
                                exported,
                            });
                        } else if let Some(specifier) = default_imports.get(&local) {
                            liveness.reexports.push(ReExport {
                                specifier: specifier.clone(),
                                imported: "default".to_string(),
                                exported,
                            });
                        }
                    }
                }
            }
            Statement::ExportDefaultDeclaration(_) => {
                liveness.exports.push("default".to_string());
            }
            Statement::ExportAllDeclaration(export) => {
                let specifier = export.source.value.to_string();
                if let Some(exported) = &export.exported {
                    // `export * as ns from S` — the whole namespace of S under one
                    // export name.
                    let name = exported.name().to_string();
                    liveness.exports.push(name.clone());
                    liveness.reexports.push(ReExport {
                        specifier,
                        imported: "*".to_string(),
                        exported: name,
                    });
                } else {
                    liveness.star_reexports.push(specifier);
                }
            }
            _ => {}
        }
    }

    liveness.body_uses = body_all
        .iter()
        .map(|specifier| BodyUse {
            specifier: specifier.clone(),
            all: true,
            names: body_names
                .get(specifier)
                .map(|names| names.iter().cloned().collect())
                .unwrap_or_default(),
        })
        .chain(
            body_names
                .iter()
                .filter(|(specifier, _)| !body_all.contains(*specifier))
                .map(|(specifier, names)| BodyUse {
                    specifier: specifier.clone(),
                    all: false,
                    names: names.iter().cloned().collect(),
                }),
        )
        .collect();
    liveness.exports.sort();
    liveness.exports.dedup();
    liveness
}

/// Collects the symbols referenced in real module code, deliberately skipping
/// the `local` names inside a bare `export { ... }` specifier list (those are
/// pure forwarding, tracked as re-exports, not body uses). An inline
/// `export const x = expr` IS body code, so its initializer is still visited.
struct BodyUseCollector<'s> {
    scoping: &'s Scoping,
    used: std::collections::HashSet<SymbolId>,
    /// Specifiers reached by a CommonJS `require("literal")` anywhere in the body.
    requires: std::collections::BTreeSet<String>,
}

impl<'a> oxc_ast_visit::Visit<'a> for BodyUseCollector<'_> {
    fn visit_export_named_declaration(
        &mut self,
        declaration: &oxc_ast::ast::ExportNamedDeclaration<'a>,
    ) {
        // Only the inline declaration (if any) is body code; the `specifiers`
        // list is export forwarding and must not count as a body use.
        if let Some(inner) = &declaration.declaration {
            self.visit_declaration(inner);
        }
    }

    fn visit_call_expression(&mut self, expression: &oxc_ast::ast::CallExpression<'a>) {
        // `require("x")` hands back the WHOLE of `x`'s `module.exports`. CommonJS has
        // no static named-export demand to record, so the only honest demand is the
        // full namespace — which is also what makes the module live.
        if let Some(literal) = expression.common_js_require() {
            self.requires.insert(literal.value.to_string());
        }
        oxc_ast_visit::walk::walk_call_expression(self, expression);
    }

    fn visit_identifier_reference(&mut self, identifier: &oxc_ast::ast::IdentifierReference<'a>) {
        if let Some(reference_id) = identifier.reference_id.get()
            && let Some(symbol) = self.scoping.get_reference(reference_id).symbol_id()
        {
            self.used.insert(symbol);
        }
    }
}

#[derive(Debug, Clone)]
enum ImportBinding {
    Namespace(String),
    Named { namespace: String, name: String },
}

/// What the build wants a module's source map to say: which file the map names,
/// which TEXT its positions refer to, and (for a generated source) that text.
pub(crate) struct MapRequest<'a> {
    pub path: &'a Path,
    pub origin: MapOrigin,
    pub source_text: Option<Arc<str>>,
}

/// Everything [`lower_module_ast`] produced.
struct LoweredModule {
    code: String,
    is_esm: bool,
    dependencies: Vec<String>,
    dependency_demands: Vec<DependencyDemand>,
    flat_module: Option<FlatModule>,
    map: Option<ModuleSourceMap>,
    /// Why no map was produced even though one was asked for. Reported as a
    /// non-fatal diagnostic rather than silently dropped, and never replaced by
    /// a guessed map.
    map_problem: Option<String>,
}

fn lower_module_ast<'a>(
    allocator: &'a Allocator,
    program: &mut oxc_ast::ast::Program<'a>,
    scoping: &Scoping,
    map_request: Option<&MapRequest<'_>>,
) -> LoweredModule {
    let dependencies = collect_dependencies(program);
    let dynamic_dependencies = collect_dynamic_dependencies(program);
    let optional_dependencies = crate::parser::collect_optional_dependencies(program);
    let dependency_syntax = crate::parser::collect_dependency_syntax(program);
    let eager_dependencies = crate::parser::collect_eager_dependencies(program);
    let mut dependency_demands = dependencies
        .iter()
        .map(|specifier| {
            (
                specifier.clone(),
                DependencyDemand {
                    specifier: specifier.clone(),
                    all: true,
                    names: Vec::new(),
                    dynamic: dynamic_dependencies.contains(specifier),
                    optional: optional_dependencies.contains(specifier),
                    require_syntax: dependency_syntax.require.contains(specifier),
                    import_syntax: dependency_syntax.import.contains(specifier),
                    eager: eager_dependencies.contains(specifier),
                },
            )
        })
        .collect::<HashMap<_, _>>();
    let is_esm = program.body.iter().any(|statement| {
        matches!(
            statement,
            Statement::ImportDeclaration(_)
                | Statement::ExportNamedDeclaration(_)
                | Statement::ExportDefaultDeclaration(_)
                | Statement::ExportAllDeclaration(_)
        )
    });

    let mut binding_expressions = HashMap::<SymbolId, ImportBinding>::new();
    let mut named_expressions = HashMap::<String, String>::new();
    let mut preamble_declarations = String::new();
    let mut preamble_exports = String::new();
    let mut import_index = 0_usize;
    let mut default_index = 0_usize;

    // A single specifier can be imported by several `import` statements in one
    // module — a source transform may inject another named import beside one
    // already present in the original source.
    // The recorded demand must be the UNION of every statement's named imports,
    // so the initial `all: true` default is downgraded (and any stale names
    // cleared) exactly once per specifier; later statements only accumulate.
    let mut demand_downgraded = std::collections::HashSet::<String>::new();
    if is_esm {
        for statement in &program.body {
            match statement {
                Statement::ImportDeclaration(declaration) => {
                    let source = declaration.source.value.to_string();
                    let demand = dependency_demands.entry(source.clone()).or_default();
                    demand.specifier = source.clone();
                    // This statement IS the ESM syntax, so the flag holds whether or
                    // not the specifier scan reached it.
                    demand.import_syntax = true;
                    // A specifier that is ALSO `require`d anywhere in this module
                    // keeps its whole-module demand: `require("m")` hands out
                    // `module.exports` wholesale, so downgrading to this import
                    // statement's named list would shake off exports the require
                    // observably reads (that is how the entry's lazy island pins —
                    // require thunks beside a named import of the same module —
                    // lost `control-boundary`'s default export and broke hydration).
                    if demand_downgraded.insert(source) && !demand.require_syntax {
                        demand.all = false;
                        demand.names.clear();
                    }
                    let Some(specifiers) = &declaration.specifiers else {
                        continue;
                    };
                    if specifiers.is_empty() {
                        continue;
                    }
                    let namespace = format!("__diffpack_import_{import_index}");
                    import_index += 1;
                    preamble_declarations.push_str(&format!("let {namespace};\n"));
                    for specifier in specifiers {
                        if !scoping
                            .get_resolved_reference_ids(specifier.local().symbol_id())
                            .is_empty()
                        {
                            match specifier {
                                ImportDeclarationSpecifier::ImportDefaultSpecifier(_) => {
                                    demand.names.push("default".into());
                                }
                                ImportDeclarationSpecifier::ImportNamespaceSpecifier(_) => {
                                    demand.all = true;
                                }
                                ImportDeclarationSpecifier::ImportSpecifier(specifier) => {
                                    demand.names.push(specifier.imported.name().to_string());
                                }
                            }
                        }
                        let (local, binding, expression) = match specifier {
                            ImportDeclarationSpecifier::ImportDefaultSpecifier(specifier) => {
                                let local = specifier.local.name.to_string();
                                (
                                    local,
                                    ImportBinding::Named {
                                        namespace: namespace.clone(),
                                        name: "default".into(),
                                    },
                                    format!("__import({namespace},\"default\")"),
                                )
                            }
                            ImportDeclarationSpecifier::ImportNamespaceSpecifier(specifier) => {
                                let local = specifier.local.name.to_string();
                                (
                                    local,
                                    ImportBinding::Namespace(namespace.clone()),
                                    namespace.clone(),
                                )
                            }
                            ImportDeclarationSpecifier::ImportSpecifier(specifier) => {
                                let local = specifier.local.name.to_string();
                                let imported = specifier.imported.name().to_string();
                                (
                                    local,
                                    ImportBinding::Named {
                                        namespace: namespace.clone(),
                                        name: imported.clone(),
                                    },
                                    format!("__import({namespace},{})", quote(&imported)),
                                )
                            }
                        };
                        binding_expressions.insert(specifier.local().symbol_id(), binding);
                        named_expressions.insert(local, expression);
                    }
                }
                Statement::ExportNamedDeclaration(declaration) => {
                    if let Some(source) = &declaration.source {
                        let key = source.value.to_string();
                        let demand = dependency_demands.entry(key.clone()).or_default();
                        demand.specifier = key.clone();
                        demand.import_syntax = true;
                        if demand_downgraded.insert(key) {
                            demand.all = false;
                        }
                        demand.names.extend(
                            declaration
                                .specifiers
                                .iter()
                                .map(|specifier| specifier.local.name().to_string()),
                        );
                    }
                    if let Some(inner) = &declaration.declaration {
                        inner.bound_names(&mut |identifier| {
                            preamble_exports
                                .push_str(&export_getter(&identifier.name, &identifier.name));
                        });
                    } else if declaration.source.is_some() {
                        let namespace = format!("__diffpack_reexport_{import_index}");
                        import_index += 1;
                        preamble_declarations.push_str(&format!("let {namespace};\n"));
                        for specifier in &declaration.specifiers {
                            preamble_exports.push_str(&export_getter(
                                &specifier.exported.name(),
                                &format!(
                                    "__import({namespace},{})",
                                    quote(&specifier.local.name())
                                ),
                            ));
                        }
                    } else {
                        for specifier in &declaration.specifiers {
                            let local = specifier.local.name();
                            let expression = named_expressions
                                .get(local.as_ref())
                                .map_or(local.as_ref(), String::as_str);
                            preamble_exports
                                .push_str(&export_getter(&specifier.exported.name(), expression));
                        }
                    }
                }
                Statement::ExportDefaultDeclaration(declaration) => {
                    let local = match &declaration.declaration {
                        ExportDefaultDeclarationKind::FunctionDeclaration(function)
                            if function.id.is_some() =>
                        {
                            function.id.as_ref().unwrap().name.to_string()
                        }
                        ExportDefaultDeclarationKind::ClassDeclaration(class)
                            if class.id.is_some() =>
                        {
                            class.id.as_ref().unwrap().name.to_string()
                        }
                        _ => {
                            let local = format!("__diffpack_default_{default_index}");
                            default_index += 1;
                            local
                        }
                    };
                    preamble_exports.push_str(&export_getter("default", &local));
                }
                Statement::ExportAllDeclaration(declaration) => {
                    let demand = dependency_demands
                        .entry(declaration.source.value.to_string())
                        .or_default();
                    demand.specifier = declaration.source.value.to_string();
                    demand.import_syntax = true;
                    demand.all = true;
                }
                _ => {}
            }
        }
    }

    let flat_module = build_flat_module(program, &dependencies, &dynamic_dependencies);

    AstModuleRewriter {
        builder: AstBuilder::new(allocator),
        scoping,
        bindings: &binding_expressions,
    }
    .visit_program(program);

    // Each AST fragment below is printed by its OWN `Codegen` and the module's
    // text is their concatenation. At the top level that is byte-for-byte what a
    // single shared printer produces — every fragment starts on a fresh line, and
    // the only printer state that survives a statement boundary
    // (`needs_semicolon`) is set exclusively in minified output, which this is
    // not. What it buys is the thing a truthful source map needs and a shared
    // printer cannot give: the exact position every fragment occupies in the
    // lowered module, so the printer's REAL token positions (taken from one
    // `Codegen::build` over the same program, below) can be placed where the
    // text actually landed instead of guessed at.
    let mut code = String::new();
    // The generated position the next fragment will start at.
    let mut generated = TextCursor::default();
    // Where each body statement's AST-printed text landed. Empty unless a map
    // was requested.
    let mut placements: Vec<FragmentPlacement> = Vec::new();
    if is_esm {
        generated.push(
            &mut code,
            "exports=module.exports=__esmNamespace();\nObject.defineProperty(exports,\"__esModule\",{value:true});\n",
        );
        generated.push(&mut code, &preamble_declarations);
        generated.push(&mut code, &preamble_exports);
    }

    // ESM INSTANTIATION ORDER. Every module this one REQUESTS — `import`,
    // `export … from`, `export *` — is evaluated, and every imported binding
    // initialized, BEFORE this module's body runs. Import declarations are hoisted
    // by the language; their position in the source says nothing about when they
    // take effect.
    //
    // Lowering each request in place broke that: a body statement written ABOVE an
    // import read its binding as `undefined`. Babel's JSX-pragma output does exactly
    // this — `var __jsx = React.createElement;` is emitted above `import React from
    // a package require assigned to a local binding — and it failed with
    // `TypeError: Cannot convert undefined or null to object` deep inside a render,
    // pointing at code that is correct ESM.
    //
    // So the request lowerings are collected in SOURCE ORDER (which is the evaluation
    // order the spec prescribes) and emitted as one prologue, ahead of the body. Each
    // request site still prints a line, so the module's generated line count — and
    // with it every source-map span — is unchanged.
    import_index = 0;
    let mut hoisted_requests = String::new();
    let mut is_request = Vec::with_capacity(program.body.len());
    for statement in &program.body {
        let request_line = match statement {
            Statement::ImportDeclaration(declaration) => {
                let request = quote(&declaration.source.value);
                let has_bindings = declaration
                    .specifiers
                    .as_ref()
                    .is_some_and(|specifiers| !specifiers.is_empty());
                if has_bindings {
                    let namespace = format!("__diffpack_import_{import_index}");
                    import_index += 1;
                    Some(format!(
                        "/*__diffpack_import:{request}__*/{namespace}=require.esm({request});\n"
                    ))
                } else {
                    Some(format!(
                        "/*__diffpack_import:{request}__*/require({request});\n"
                    ))
                }
            }
            Statement::ExportNamedDeclaration(declaration)
                if declaration.declaration.is_none() && declaration.source.is_some() =>
            {
                let request = quote(&declaration.source.as_ref().expect("checked above").value);
                let namespace = format!("__diffpack_reexport_{import_index}");
                import_index += 1;
                Some(format!(
                    "/*__diffpack_import:{request}__*/{namespace}=require.esm({request});\n"
                ))
            }
            Statement::ExportAllDeclaration(declaration) => {
                let request = quote(&declaration.source.value);
                Some(match &declaration.exported {
                    Some(exported) => {
                        export_getter(&exported.name(), &format!("require.esm({request})"))
                    }
                    None => format!("__reExport(exports,require.esm({request}));\n"),
                })
            }
            _ => None,
        };
        is_request.push(request_line.is_some());
        if let Some(line) = request_line {
            hoisted_requests.push_str(&line);
        }
    }
    generated.push(&mut code, &hoisted_requests);

    // The reference print of every body statement: what Oxc's own
    // `Program` printer emits for it. Only needed to align that printer's source
    // map onto the lowered text, so it is skipped entirely when no map is wanted.
    let reference_fragments: Vec<String> = if map_request.is_some() {
        program.body.iter().map(print_statement_text).collect()
    } else {
        Vec::new()
    };

    default_index = 0;
    for (index, (statement, hoisted)) in program.body.iter().zip(&is_request).enumerate() {
        if *hoisted {
            // Already emitted in the prologue above; keep the line so the module's
            // line count (and every source-map span derived from it) is unchanged.
            generated.push(&mut code, "\n");
            continue;
        }
        match statement {
            Statement::ExportNamedDeclaration(declaration) => {
                if let Some(inner) = &declaration.declaration {
                    let mut names = Vec::new();
                    inner.bound_names(&mut |identifier| names.push(identifier.name.to_string()));
                    // Every obviously-pure declaration is marked removable; the
                    // emit-time shake decides liveness transitively (demand +
                    // references from retained code), so "locally used by other
                    // dead code" no longer pins a declaration.
                    let removable = declaration_is_obviously_pure(inner);
                    if removable && !names.is_empty() {
                        generated.push(
                            &mut code,
                            &format!("/*__diffpack_decl:{}__*/\n", names.join(",")),
                        );
                    }
                    // The lowering emits the DECLARATION alone where the reference
                    // print emits the whole export statement. What it left out is
                    // measured from the two texts (see `reference_prefix`), never
                    // assumed: it is `export ` here, but an annotation the printer
                    // puts on a line of its own (`/* @__NO_SIDE_EFFECTS__ */`)
                    // makes the reference a whole LINE taller, and a fixed column
                    // delta silently placed every token of such a statement one
                    // generated line too low.
                    let start = generated.position();
                    let start_byte = code.len();
                    generated.push(&mut code, &print_declaration_text(inner));
                    placements.push(FragmentPlacement {
                        statement: index,
                        generated: start,
                        aligned: start_byte..code.len(),
                        dropped: "export ",
                    });
                    if removable && !names.is_empty() {
                        generated.push(&mut code, "/*__diffpack_decl_end__*/\n");
                    }
                }
            }
            Statement::ExportDefaultDeclaration(declaration) => {
                let is_named = matches!(
                    &declaration.declaration,
                    ExportDefaultDeclarationKind::FunctionDeclaration(function)
                        if function.id.is_some()
                ) || matches!(
                    &declaration.declaration,
                    ExportDefaultDeclarationKind::ClassDeclaration(class)
                        if class.id.is_some()
                );
                if !is_named {
                    generated.push(
                        &mut code,
                        &format!("const __diffpack_default_{default_index}="),
                    );
                    default_index += 1;
                }
                let start = generated.position();
                let start_byte = code.len();
                generated.push(&mut code, &print_default_text(&declaration.declaration));
                placements.push(FragmentPlacement {
                    statement: index,
                    generated: start,
                    aligned: start_byte..code.len(),
                    dropped: "export default ",
                });
                generated.push(&mut code, "\n");
            }
            _ => {
                // A plain (non-exported) pure top-level declaration is also
                // removable: a helper only dead exports referenced must fall
                // with them. Impure statements print unmarked and anchor the
                // shake's live set.
                let removable_names = statement.as_declaration().and_then(|declaration| {
                    if !declaration_is_obviously_pure(declaration) {
                        return None;
                    }
                    let mut names = Vec::new();
                    declaration
                        .bound_names(&mut |identifier| names.push(identifier.name.to_string()));
                    (!names.is_empty()).then_some(names)
                });
                if let Some(names) = &removable_names {
                    generated.push(
                        &mut code,
                        &format!("/*__diffpack_decl:{}__*/\n", names.join(",")),
                    );
                }
                let start = generated.position();
                let start_byte = code.len();
                match reference_fragments.get(index) {
                    // The lowering prints this statement exactly as the reference
                    // print does, so reuse the string instead of printing twice.
                    Some(fragment) => generated.push(&mut code, fragment),
                    None => generated.push(&mut code, &print_statement_text(statement)),
                }
                placements.push(FragmentPlacement {
                    statement: index,
                    generated: start,
                    aligned: start_byte..code.len(),
                    dropped: "",
                });
                generated.push(&mut code, "\n");
                if removable_names.is_some() {
                    generated.push(&mut code, "/*__diffpack_decl_end__*/\n");
                }
            }
        }
    }
    if is_esm {
        generated.push(&mut code, "__seal(exports);");
    }
    let mut dependency_demands = dependency_demands.into_values().collect::<Vec<_>>();
    for demand in &mut dependency_demands {
        demand.names.sort();
        demand.names.dedup();
    }
    dependency_demands.sort_by(|left, right| left.specifier.cmp(&right.specifier));

    let (map, map_problem) = match map_request {
        Some(request) => {
            match build_module_map(
                allocator,
                program,
                request,
                &reference_fragments,
                &placements,
                &code,
            ) {
                Ok(map) => (Some(map), None),
                Err(problem) => (None, Some(problem)),
            }
        }
        None => (None, None),
    };

    let flat_module = flat_module.map(|mut flat| {
        let (flat_code, flat_lines) =
            derive_flat_code(&code, &flat.import_replacements, map.is_some());
        flat.code = flat_code;
        flat.map_lines = flat_lines;
        flat
    });
    LoweredModule {
        code,
        is_esm,
        dependencies,
        dependency_demands,
        flat_module,
        map,
        map_problem,
    }
}

/// A running (line, column) position in generated text, in the units a source map
/// speaks: lines counted from 0, columns in UTF-16 code units.
#[derive(Clone, Copy, Debug, Default)]
struct TextCursor {
    line: u32,
    column: u32,
}

impl TextCursor {
    fn position(self) -> (u32, u32) {
        (self.line, self.column)
    }

    /// Append `text` and advance past it.
    fn push(&mut self, buffer: &mut String, text: &str) {
        buffer.push_str(text);
        self.advance(text);
    }

    fn advance(&mut self, text: &str) {
        match text.rfind('\n') {
            Some(last) => {
                self.line += text.bytes().filter(|byte| *byte == b'\n').count() as u32;
                self.column = crate::source_map::utf16_len(&text[last + 1..]);
            }
            None => self.column += crate::source_map::utf16_len(text),
        }
    }
}

/// Where one body statement's AST-printed text landed in the lowered module.
struct FragmentPlacement {
    /// Index into `program.body`.
    statement: usize,
    /// Position in the lowered module where the AST-printed text starts.
    generated: (u32, u32),
    /// Byte range of that text inside the lowered module.
    aligned: std::ops::Range<usize>,
    /// The keyword the lowering emitted this statement WITHOUT, which the
    /// reference printer emits (`export `, `export default `, or nothing). The
    /// rest of the difference between the two prints is measured from the texts
    /// themselves — see [`align_fragment`].
    dropped: &'static str,
}

/// How a statement's reference print lines up with the text the lowering emitted
/// for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FragmentAlignment {
    /// Position in the reference print where the lowering's text begins. Anything
    /// before it is text the lowering replaced.
    skip: (u32, u32),
    /// Position in the reference print where the two prints stop agreeing. Beyond
    /// it nothing can be placed, because the lowering's text has diverged.
    limit: (u32, u32),
}

/// Line up a statement's reference print with the text the lowering emitted for
/// it, so the printer's positions can be moved from one onto the other.
///
/// The two prints differ at both ends and neither difference may be assumed:
///
/// * At the FRONT the lowering leaves out `dropped` (`export ` /
///   `export default `) — and the reference printer may put an annotation comment
///   (`/* @__NO_SIDE_EFFECTS__ */`) ahead of that, on a line of its own. Modelling
///   the front as a fixed COLUMN delta was wrong the moment it spanned a line:
///   every token of an annotated exported declaration was placed one generated
///   line too low, silently, and the map was emitted anyway.
/// * At the BACK they can diverge outright: the lowering prints `export const X =
///   3` as `const X = 3` with no terminating `;`, which the statement printer
///   emits. So agreement is measured, and a token past the point where the two
///   texts stop agreeing is not placed at all.
///
/// `None` when the front cannot be identified — the keyword is missing, or
/// something other than whitespace and annotation comments precedes it. Then this
/// function does not know what the lowering replaced, and the caller must not
/// guess.
fn align_fragment(reference: &str, dropped: &str, aligned: &str) -> Option<FragmentAlignment> {
    let offset = if dropped.is_empty() {
        0
    } else {
        let keyword = reference.find(dropped)?;
        // Everything the reference printed ahead of the keyword has to be text a
        // printer emits on its own account — whitespace and annotation comments.
        // Anything else means the keyword found is not the one that was dropped.
        if !is_annotation_prefix(&reference[..keyword]) {
            return None;
        }
        keyword + dropped.len()
    };
    let tail = reference.get(offset..)?;
    let mut agree = tail
        .bytes()
        .zip(aligned.bytes())
        .take_while(|(left, right)| left == right)
        .count();
    while agree > 0 && !tail.is_char_boundary(agree) {
        agree -= 1;
    }
    let mut cursor = TextCursor::default();
    cursor.advance(&reference[..offset]);
    let skip = cursor.position();
    cursor.advance(&tail[..agree]);
    Some(FragmentAlignment {
        skip,
        limit: cursor.position(),
    })
}

/// Whether `text` is only what a printer emits ahead of a statement's keyword:
/// whitespace and block comments (the `/* @__PURE__ */` /
/// `/* @__NO_SIDE_EFFECTS__ */` annotations Oxc prints from AST flags). Source
/// comments never reach these prints — the printers that produce them have no
/// comment map.
fn is_annotation_prefix(text: &str) -> bool {
    let mut rest = text.trim_start();
    while let Some(body) = rest.strip_prefix("/*") {
        let Some(end) = body.find("*/") else {
            return false;
        };
        rest = body[end + 2..].trim_start();
    }
    rest.is_empty()
}

/// Print one statement exactly as the lowering emits it.
fn print_statement_text(statement: &Statement<'_>) -> String {
    let mut codegen = Codegen::new();
    statement.print(&mut codegen, Context::default());
    codegen.into_source_text()
}

fn print_declaration_text(declaration: &oxc_ast::ast::Declaration<'_>) -> String {
    let mut codegen = Codegen::new();
    print_declaration(&mut codegen, declaration);
    codegen.into_source_text()
}

fn print_default_text(declaration: &ExportDefaultDeclarationKind<'_>) -> String {
    let mut codegen = Codegen::new();
    declaration.print(&mut codegen, Context::default());
    codegen.into_source_text()
}

/// Turn the Oxc printer's own source map for `program` into a map over the
/// LOWERED module text.
///
/// The printer will only emit a map from [`Codegen::build`], which prints a whole
/// `Program` — it cannot be driven statement by statement. So the program is
/// printed once more, in full, purely for its map, and that map's tokens are
/// moved onto the lowered text using the per-statement placements the lowering
/// recorded. The move is only legitimate if the reference print really is the
/// concatenation of the same per-statement prints the lowering used, so that is
/// CHECKED, byte for byte, before a single token is emitted: a mismatch produces
/// no map at all rather than a plausible-looking wrong one.
fn build_module_map<'a>(
    allocator: &'a Allocator,
    program: &mut Program<'a>,
    request: &MapRequest<'_>,
    reference_fragments: &[String],
    placements: &[FragmentPlacement],
    code: &str,
) -> Result<ModuleSourceMap, String> {
    // The lowering prints no comments, no directives and no hashbang: its
    // per-statement printers have an empty comment map and never see them. Clear
    // them here so the reference print is driven by exactly the same state.
    // `program` is dead after this function, so nothing downstream can notice.
    let builder = AstBuilder::new(allocator);
    program.comments = oxc_allocator::Vec::new_in(&builder);
    program.directives = oxc_allocator::Vec::new_in(&builder);
    program.hashbang = None;
    // A leading string-literal expression statement is the one statement Oxc's
    // `Program` printer parenthesizes (so it cannot be mistaken for a directive)
    // and the lowering does not. An empty statement in front of it takes that
    // role, keeping every real statement's reference print equal to the
    // lowering's. Its span is the end of the source, a position no real node
    // starts at, so its own mapping token is unmistakable and is discarded with
    // the rest of its fragment.
    let end_of_source = u32::try_from(program.source_text.len()).unwrap_or(u32::MAX);
    let guard =
        Statement::new_empty_statement(oxc_span::Span::new(end_of_source, end_of_source), &builder);
    let guard_text = print_statement_text(&guard);
    program.body.insert(0, guard);
    let options = CodegenOptions {
        source_map_path: Some(request.path.to_path_buf()),
        ..CodegenOptions::default()
    };
    let printed = Codegen::new().with_options(options).build(program);
    program.body.remove(0);

    // The reference print must be exactly the concatenation of the per-statement
    // prints, or the tokens cannot be attributed to statements at all. Compared
    // fragment by fragment against slices of the print rather than by building the
    // concatenation: this runs on every module of the build, in parallel across
    // every core, and materializing a second copy of each module's printed text
    // costs peak memory for nothing.
    let mut consumed = 0_usize;
    let mut matches = |fragment: &str| {
        let end = consumed + fragment.len();
        let agrees = printed.code.get(consumed..end) == Some(fragment);
        consumed = end;
        agrees
    };
    let concatenates = matches(&guard_text)
        && reference_fragments.iter().all(|fragment| matches(fragment))
        && consumed == printed.code.len();
    if !concatenates {
        return Err(
            "the reference print of this module is not the concatenation of its statements, \
             so the printer's positions cannot be placed in the lowered text"
                .to_string(),
        );
    }
    let map = printed
        .map
        .ok_or_else(|| "the printer produced no source map".to_string())?;

    // Where each reference fragment starts inside the reference print.
    let mut starts = Vec::with_capacity(reference_fragments.len() + 1);
    let mut cursor = TextCursor::default();
    cursor.advance(&guard_text);
    for fragment in reference_fragments {
        starts.push(cursor.position());
        cursor.advance(fragment);
    }
    starts.push(cursor.position());

    let names: Vec<String> = map.get_names().map(str::to_owned).collect();
    // Tokens come out of the printer in generated order, and the placements are in
    // statement order, so each fragment's tokens are one contiguous run found by
    // binary search — never a scan per statement.
    let printed_tokens: Vec<oxc_sourcemap::Token> = map.get_tokens().collect();
    let mut tokens = Vec::new();
    for placement in placements {
        let Some(&(start_line, start_column)) = starts.get(placement.statement) else {
            continue;
        };
        let (end_line, end_column) = starts[placement.statement + 1];
        let (lowered_line, lowered_column) = placement.generated;
        // Where the text the lowering emitted for this statement sits inside the
        // statement's reference print — measured, not assumed. A statement whose
        // front cannot be identified gets no map at all (for the whole module),
        // because the alternative is emitting positions that are wrong by however
        // much the two prints differ.
        let reference = reference_fragments
            .get(placement.statement)
            .ok_or_else(|| format!("statement {} has no reference print", placement.statement))?;
        let aligned = code.get(placement.aligned.clone()).ok_or_else(|| {
            format!(
                "statement {}'s lowered text is not a range of the lowered module",
                placement.statement
            )
        })?;
        let alignment = align_fragment(reference, placement.dropped, aligned).ok_or_else(|| {
            format!(
                "statement {} does not begin with the `{}` the lowering dropped, so the \
                     printer's positions cannot be placed in it",
                placement.statement,
                placement.dropped.trim_end()
            )
        })?;
        let (skip_line, skip_column) = alignment.skip;
        let first = printed_tokens.partition_point(|token| {
            (token.get_dst_line(), token.get_dst_col()) < (start_line, start_column)
        });
        let last = printed_tokens.partition_point(|token| {
            (token.get_dst_line(), token.get_dst_col()) < (end_line, end_column)
        });
        for token in &printed_tokens[first..last] {
            let line = token.get_dst_line();
            let column = token.get_dst_col();
            let local_line = line - start_line;
            let local_column = if local_line == 0 {
                column - start_column
            } else {
                column
            };
            if (local_line, local_column) > alignment.limit {
                // Past where the two prints agree — the lowering's text has
                // diverged from the reference print (a statement terminator it
                // does not emit), so there is no position here to map onto.
                continue;
            }
            let (generated_line, generated_column) =
                if (local_line, local_column) < (skip_line, skip_column) {
                    // Inside the text the lowering replaced: the `export` /
                    // `export default` keyword, and any annotation comment the
                    // reference printer put ahead of it (possibly on its own line).
                    // The token still marks where this statement begins, and in the
                    // lowered text that is the start of its first line.
                    (lowered_line, 0)
                } else if local_line == skip_line {
                    (lowered_line, lowered_column + (local_column - skip_column))
                } else {
                    // Past the fragment's first line the lowering copied the reference
                    // print's lines verbatim, each starting at column 0 in both texts.
                    (lowered_line + (local_line - skip_line), local_column)
                };
            // A node the lowering SYNTHESIZED carries a zero span, which the
            // printer resolves to line 1, column 0 — a position the code never
            // came from. Verified empirically: Oxc's own TS/JSX lowering keeps the
            // original spans, and diffpack's module rewriter now carries the
            // replaced reference's span, so the only tokens left here at the very
            // first byte are synthesized ones. That single position is given up
            // rather than risk attributing generated code to the file's first
            // line.
            if (token.get_src_line(), token.get_src_col()) == (0, 0) {
                continue;
            }
            tokens.push(MapToken {
                generated_line,
                generated_column,
                source_line: token.get_src_line(),
                source_column: token.get_src_col(),
                name: token.get_name_id(),
            });
        }
    }
    Ok(ModuleSourceMap::new(
        request.origin,
        request.source_text.clone(),
        names,
        tokens,
        code.lines().count().max(1),
    ))
}

fn print_declaration(codegen: &mut Codegen<'_>, declaration: &oxc_ast::ast::Declaration<'_>) {
    match declaration {
        oxc_ast::ast::Declaration::VariableDeclaration(declaration) => {
            declaration.print(codegen, Context::default());
        }
        oxc_ast::ast::Declaration::FunctionDeclaration(declaration) => {
            declaration.print(codegen, Context::default());
        }
        oxc_ast::ast::Declaration::ClassDeclaration(declaration) => {
            declaration.print(codegen, Context::default());
        }
        _ => {}
    }
    codegen.print_str("\n");
}

fn build_flat_module(
    program: &oxc_ast::ast::Program<'_>,
    dependencies: &[String],
    dynamic_dependencies: &std::collections::BTreeSet<String>,
) -> Option<FlatModule> {
    let foldable = build_foldable_module(program);
    let mut static_imports = Vec::new();
    let mut declarations = Vec::new();
    let mut exports = Vec::new();
    let mut has_direct_effects = false;
    let mut import_replacements = Vec::new();
    let mut binding_import_index = 0_usize;

    for statement in &program.body {
        match statement {
            Statement::ImportDeclaration(import) => {
                static_imports.push(import.source.value.to_string());
                if let Some(specifiers) = &import.specifiers {
                    let has_bindings = !specifiers.is_empty();
                    for specifier in specifiers {
                        match specifier {
                            ImportDeclarationSpecifier::ImportSpecifier(specifier)
                                if specifier.imported.name() == specifier.local.name =>
                            {
                                import_replacements.push((
                                    format!("__diffpack_import_{binding_import_index}"),
                                    specifier.imported.name().to_string(),
                                ));
                            }
                            _ => return None,
                        }
                    }
                    if has_bindings {
                        binding_import_index += 1;
                    }
                }
            }
            Statement::ExportNamedDeclaration(export) if export.source.is_none() => {
                if let Some(declaration) = &export.declaration {
                    let mut names = Vec::new();
                    declaration.bound_names(&mut |identifier| {
                        names.push(identifier.name.to_string());
                    });
                    declarations.extend(names.iter().cloned());
                    exports.extend(names.iter().cloned());
                    has_direct_effects |= !declaration_is_obviously_pure(declaration);
                } else {
                    for specifier in &export.specifiers {
                        if specifier.local.name() != specifier.exported.name() {
                            return None;
                        }
                        exports.push(specifier.exported.name().to_string());
                    }
                }
            }
            Statement::ExportNamedDeclaration(_)
            | Statement::ExportDefaultDeclaration(_)
            | Statement::ExportAllDeclaration(_) => return None,
            Statement::VariableDeclaration(declaration) => {
                declaration.bound_names(&mut |identifier| {
                    declarations.push(identifier.name.to_string());
                });
                has_direct_effects |= declaration.declarations.iter().any(|declarator| {
                    declarator
                        .init
                        .as_ref()
                        .is_some_and(|init| !expression_is_obviously_pure(init))
                });
            }
            Statement::FunctionDeclaration(declaration) => {
                declaration.bound_names(&mut |identifier| {
                    declarations.push(identifier.name.to_string());
                });
            }
            Statement::ClassDeclaration(declaration) => {
                declaration.bound_names(&mut |identifier| {
                    declarations.push(identifier.name.to_string());
                });
                has_direct_effects = true;
            }
            _ => {
                has_direct_effects = true;
            }
        }
    }
    if dependencies.iter().any(|dependency| {
        !static_imports.contains(dependency) && !dynamic_dependencies.contains(dependency)
    }) {
        return None;
    }
    declarations.sort();
    declarations.dedup();
    exports.sort();
    exports.dedup();
    Some(FlatModule {
        code: String::new(),
        map_lines: None,
        declarations,
        exports,
        has_direct_effects,
        import_replacements,
        foldable,
    })
}

fn build_foldable_module(program: &oxc_ast::ast::Program<'_>) -> Option<FoldableModule> {
    let mut module = FoldableModule::default();
    for statement in &program.body {
        match statement {
            Statement::ImportDeclaration(_) => {}
            Statement::ExportNamedDeclaration(export) if export.source.is_none() => {
                let Declaration::VariableDeclaration(declaration) = export.declaration.as_ref()?
                else {
                    return None;
                };
                if declaration.kind != VariableDeclarationKind::Const {
                    return None;
                }
                for declarator in &declaration.declarations {
                    let BindingPattern::BindingIdentifier(identifier) = &declarator.id else {
                        return None;
                    };
                    module.constants.push((
                        identifier.name.to_string(),
                        fold_expression(declarator.init.as_ref()?)?,
                    ));
                }
            }
            Statement::ExpressionStatement(statement) => {
                let Expression::CallExpression(call) = &statement.expression else {
                    return None;
                };
                let Expression::StaticMemberExpression(member) = &call.callee else {
                    return None;
                };
                let Expression::Identifier(object) = &member.object else {
                    return None;
                };
                if object.name != "console"
                    || member.property.name != "log"
                    || call.arguments.len() != 1
                {
                    return None;
                }
                module
                    .console_logs
                    .push(fold_expression(call.arguments[0].as_expression()?)?);
            }
            Statement::EmptyStatement(_) => {}
            _ => return None,
        }
    }
    Some(module)
}

fn fold_expression(expression: &Expression<'_>) -> Option<FoldExpression> {
    match expression {
        Expression::NumericLiteral(number) => Some(FoldExpression::Number(number.value.to_bits())),
        Expression::Identifier(identifier) => {
            Some(FoldExpression::Reference(identifier.name.to_string()))
        }
        Expression::BinaryExpression(binary) if binary.operator == BinaryOperator::Addition => {
            Some(FoldExpression::Add(
                Box::new(fold_expression(&binary.left)?),
                Box::new(fold_expression(&binary.right)?),
            ))
        }
        Expression::ParenthesizedExpression(parenthesized) => {
            fold_expression(&parenthesized.expression)
        }
        _ => None,
    }
}

fn derive_flat_code(
    code: &str,
    replacements: &[(String, String)],
    track_lines: bool,
) -> (String, Option<LineTrack>) {
    let mut flat = String::with_capacity(code.len());
    let mut kept: Vec<usize> = Vec::new();
    for (index, line) in code.lines().enumerate() {
        if line.starts_with("exports=module.exports=__esmNamespace()")
            || line.starts_with("Object.defineProperty(exports,\"__esModule\"")
            || line.starts_with("let __diffpack_import_")
            || line.starts_with("/*__diffpack_export:")
            || line.starts_with("/*__diffpack_import:")
            || line == "__seal(exports);"
        {
            continue;
        }
        if track_lines {
            kept.push(index);
        }
        flat.push_str(line);
        flat.push('\n');
    }
    // Every dropped line above is bundler glue, and every kept line is verbatim,
    // so the surviving lines carry their own map positions unchanged. The binding
    // replacements below DO rewrite real code in place, so each one is recorded as
    // a column edit: a token inside a replaced span is dropped, one after it moves
    // by exactly the amount the line shrank.
    //
    // All the replacements run in ONE pass. Doing them one at a time re-measured
    // each needle's columns in the text the previous replacement had already
    // rewritten, while appending every edit into the same `LineTrack` — two
    // coordinate systems in one edit list, which `LineOrigin::remap_column`
    // resolves into confident, wrong generated columns (see
    // `replace_many_tracked`). A line carrying two different import bindings is
    // the common case (`twMerge(clsx(inputs))`), so this was not an edge.
    let mut track =
        track_lines.then(|| LineTrack::identity(code.lines().count()).keep(kept.into_iter()));
    let pairs = replacements
        .iter()
        .flat_map(|(namespace, name)| {
            [
                (format!("__import({namespace}, \"{name}\")"), name.clone()),
                (format!("__import({namespace},\"{name}\")"), name.clone()),
            ]
        })
        .collect::<Vec<_>>();
    // A build with no map to carry runs the SAME single pass against a track with
    // no lines (whose `record_edit` is a no-op), so the emitted bytes cannot
    // depend on whether source maps were asked for.
    let mut untracked = LineTrack::default();
    let edits = match track.as_mut() {
        Some(track) => track,
        None => &mut untracked,
    };
    if let Some(replaced) = crate::source_map::replace_many_tracked(&flat, &pairs, edits) {
        flat = replaced;
    }
    (flat, track)
}

fn declaration_is_obviously_pure(declaration: &oxc_ast::ast::Declaration<'_>) -> bool {
    match declaration {
        oxc_ast::ast::Declaration::FunctionDeclaration(_) => true,
        oxc_ast::ast::Declaration::VariableDeclaration(declaration) => {
            declaration.declarations.iter().all(|declarator| {
                declarator
                    .init
                    .as_ref()
                    .is_none_or(expression_is_obviously_pure)
            })
        }
        _ => false,
    }
}

/// Whether evaluating `expression` can have no observable side effect, so a
/// declaration initialized by it may be dropped when nothing live references
/// it. Deliberately syntactic and conservative — anything not recognized is
/// impure. Identifier references are allowed: dropping a dead `const a = b`
/// only changes behavior for a program whose evaluation would have thrown
/// (TDZ / missing global), the same stance the reference bundlers take.
fn expression_is_obviously_pure(expression: &oxc_ast::ast::Expression<'_>) -> bool {
    use oxc_ast::ast::Expression;
    match expression {
        Expression::BooleanLiteral(_)
        | Expression::NullLiteral(_)
        | Expression::NumericLiteral(_)
        | Expression::BigIntLiteral(_)
        | Expression::StringLiteral(_)
        | Expression::RegExpLiteral(_)
        | Expression::FunctionExpression(_)
        | Expression::ArrowFunctionExpression(_)
        | Expression::Identifier(_) => true,
        Expression::TemplateLiteral(template) => template
            .expressions
            .iter()
            .all(expression_is_obviously_pure),
        Expression::ArrayExpression(array) => array.elements.iter().all(|element| match element {
            oxc_ast::ast::ArrayExpressionElement::Elision(_) => true,
            oxc_ast::ast::ArrayExpressionElement::SpreadElement(_) => false,
            element => element
                .as_expression()
                .is_some_and(expression_is_obviously_pure),
        }),
        Expression::ObjectExpression(object) => {
            object.properties.iter().all(|property| match property {
                oxc_ast::ast::ObjectPropertyKind::ObjectProperty(property) => {
                    (!property.computed
                        || property
                            .key
                            .as_expression()
                            .is_some_and(expression_is_obviously_pure))
                        && expression_is_obviously_pure(&property.value)
                }
                oxc_ast::ast::ObjectPropertyKind::SpreadProperty(_) => false,
            })
        }
        Expression::UnaryExpression(unary) => {
            unary.operator != oxc_syntax::operator::UnaryOperator::Delete
                && expression_is_obviously_pure(&unary.argument)
        }
        Expression::BinaryExpression(binary) => {
            // `in`/`instanceof` can throw on non-object operands and private
            // names; arithmetic/comparison on pure operands cannot observe.
            !matches!(
                binary.operator,
                BinaryOperator::In | BinaryOperator::Instanceof
            ) && expression_is_obviously_pure(&binary.left)
                && expression_is_obviously_pure(&binary.right)
        }
        Expression::LogicalExpression(logical) => {
            expression_is_obviously_pure(&logical.left)
                && expression_is_obviously_pure(&logical.right)
        }
        Expression::ConditionalExpression(conditional) => {
            expression_is_obviously_pure(&conditional.test)
                && expression_is_obviously_pure(&conditional.consequent)
                && expression_is_obviously_pure(&conditional.alternate)
        }
        Expression::ParenthesizedExpression(inner) => {
            expression_is_obviously_pure(&inner.expression)
        }
        _ => false,
    }
}

struct AstModuleRewriter<'a, 's> {
    builder: AstBuilder<'a>,
    scoping: &'s Scoping,
    bindings: &'s HashMap<SymbolId, ImportBinding>,
}

#[allow(deprecated)]
impl<'a> AstModuleRewriter<'a, '_> {
    /// `span` is the span of the reference being replaced. Carrying it onto the
    /// synthesized node is what lets the source map say where `__import(ns,"x")`
    /// came from: it stands for exactly that identifier. A default (zero) span
    /// would instead make the printer emit a mapping to the file's very first
    /// byte — a position the code never came from.
    fn binding_expression(
        &self,
        binding: &ImportBinding,
        span: oxc_span::Span,
    ) -> oxc_ast::ast::Expression<'a> {
        match binding {
            ImportBinding::Namespace(namespace) => self
                .builder
                .expression_identifier(span, self.builder.ident(namespace)),
            ImportBinding::Named { namespace, name } => self.call(
                "__import",
                [
                    self.builder
                        .expression_identifier(span, self.builder.ident(namespace)),
                    self.builder
                        .expression_string_literal(span, self.builder.str(name), None),
                ],
                span,
            ),
        }
    }

    fn call<const N: usize>(
        &self,
        name: &str,
        arguments: [oxc_ast::ast::Expression<'a>; N],
        span: oxc_span::Span,
    ) -> oxc_ast::ast::Expression<'a> {
        self.builder.expression_call(
            span,
            self.builder
                .expression_identifier(span, self.builder.ident(name)),
            NONE,
            self.builder
                .vec_from_iter(arguments.into_iter().map(oxc_ast::ast::Argument::from)),
            false,
        )
    }

    fn identifier_binding(
        &self,
        identifier: &oxc_ast::ast::IdentifierReference<'a>,
    ) -> Option<&ImportBinding> {
        let reference_id = identifier.reference_id.get()?;
        let symbol_id = self.scoping.get_reference(reference_id).symbol_id()?;
        self.bindings.get(&symbol_id)
    }
}

#[allow(deprecated)]
impl<'a> VisitMut<'a> for AstModuleRewriter<'a, '_> {
    fn visit_expression(&mut self, expression: &mut oxc_ast::ast::Expression<'a>) {
        if let oxc_ast::ast::Expression::Identifier(identifier) = expression
            && let Some(binding) = self.identifier_binding(identifier).cloned()
        {
            let span = identifier.span;
            *expression = self.binding_expression(&binding, span);
            return;
        }
        if let oxc_ast::ast::Expression::ImportExpression(import) = expression
            && let oxc_ast::ast::Expression::StringLiteral(literal) = &import.source
        {
            let span = import.span;
            *expression = self.call(
                "__dynamic",
                [
                    self.builder
                        .expression_identifier(span, self.builder.ident("require")),
                    self.builder.expression_string_literal(
                        span,
                        self.builder.str(&literal.value),
                        None,
                    ),
                ],
                span,
            );
            return;
        }
        walk_mut::walk_expression(self, expression);
    }

    fn visit_object_property(&mut self, property: &mut oxc_ast::ast::ObjectProperty<'a>) {
        if property.shorthand
            && let oxc_ast::ast::Expression::Identifier(identifier) = &property.value
            && self.identifier_binding(identifier).is_some()
        {
            property.shorthand = false;
        }
        walk_mut::walk_object_property(self, property);
    }
}

fn export_getter(exported: &str, expression: &str) -> String {
    format!(
        "/*__diffpack_export:{}__*/__export(exports,{},()=>{});\n",
        exported,
        quote(exported),
        expression
    )
}

fn quote(value: &str) -> String {
    serde_json::to_string(value).expect("serializing a JavaScript string cannot fail")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Transform a module with source maps on, and return the map's tokens paired
    /// with the text they claim on each side.
    fn mapped_lines(path: &str, source: &str) -> (String, Vec<(u32, u32, u32, u32)>) {
        let transformed = transform_module_in_language(
            Path::new(path),
            source,
            Target::Server,
            false,
            JsxExtensions::default(),
            &ProjectConfig::default(),
            SourceLanguage::FromPath,
            true,
        );
        assert!(
            transformed.diagnostics.is_empty(),
            "{:?}",
            transformed.diagnostics
        );
        let map = transformed
            .map
            .expect("the module must carry a map — a diagnostic says why when it does not");
        let tokens = map
            .tokens()
            .iter()
            .map(|token| {
                (
                    token.generated_line,
                    token.generated_column,
                    token.source_line,
                    token.source_column,
                )
            })
            .collect();
        (transformed.code, tokens)
    }

    /// Oxc prints `/* @__NO_SIDE_EFFECTS__ */` on a line of its OWN ahead of an
    /// exported declaration, and the lowering — which prints the declaration
    /// alone — does not. Modelling the difference between the two prints as a
    /// fixed COLUMN offset therefore put every token of such a statement one
    /// generated line too low, silently: the map was emitted, it looked plausible,
    /// and a debugger stopped one line off through the whole function.
    ///
    /// The trigger is narrow but it is real npm code (`@vitest/runner`,
    /// `@opentelemetry/semantic-conventions`, svelte's `dom/template.js`), so the
    /// alignment is measured from the printed texts and locked here.
    #[test]
    fn an_annotated_export_maps_to_the_lines_it_really_occupies() {
        let source = "export function first(a) {\n  return a + 1\n}\n\n\
                      /*#__NO_SIDE_EFFECTS__*/\n\
                      export function second(b) {\n  const SECOND = \"s\"\n  return b + SECOND\n}\n";
        let (code, tokens) = mapped_lines("annotated.js", source);
        let lines: Vec<&str> = code.lines().collect();
        let sources: Vec<&str> = source.lines().collect();

        let generated = lines
            .iter()
            .position(|line| line.starts_with("function second("))
            .expect("the lowering emits the declaration without its `export`");
        let on_line: Vec<_> = tokens
            .iter()
            .filter(|(line, ..)| *line as usize == generated)
            .collect();
        assert!(
            !on_line.is_empty(),
            "the annotated declaration's own line must be mapped, got {tokens:?}\n{code}"
        );
        for (_, _, source_line, _) in &on_line {
            assert_eq!(
                sources[*source_line as usize], "export function second(b) {",
                "the declaration must map to the line it was written on"
            );
        }
        // Its body maps to its body, not to the line above it.
        let body = tokens
            .iter()
            .find(|(line, ..)| *line as usize == generated + 1)
            .expect("the declaration's first body line must be mapped");
        assert_eq!(
            sources[body.2 as usize].trim(),
            "const SECOND = \"s\"",
            "the body line must map to the body, got {:?}",
            sources[body.2 as usize]
        );
        // Every token must name a position that exists on both sides.
        for (generated_line, generated_column, source_line, source_column) in &tokens {
            let emitted = lines
                .get(*generated_line as usize)
                .unwrap_or_else(|| panic!("generated line {generated_line} does not exist"));
            assert!(
                *generated_column as usize <= emitted.chars().count(),
                "generated column {generated_column} is past the end of {emitted:?}"
            );
            let original = sources
                .get(*source_line as usize)
                .unwrap_or_else(|| panic!("source line {source_line} does not exist"));
            assert!(
                *source_column as usize <= original.chars().count(),
                "source column {source_column} is past the end of {original:?}"
            );
        }
    }

    /// The statement printer terminates `export const X = 3` with a `;` that the
    /// lowering's declaration printer does not emit. The two prints therefore
    /// agree up to a point and then diverge, and only what agrees may be mapped.
    #[test]
    fn a_statement_whose_prints_diverge_at_the_end_still_maps_what_agrees() {
        let source = "export const FIRST = 3;\nexport const SECOND = FIRST + 1;\n";
        let (code, tokens) = mapped_lines("terminated.js", source);
        let lines: Vec<&str> = code.lines().collect();
        let sources: Vec<&str> = source.lines().collect();
        assert!(
            !tokens.is_empty(),
            "a module of plain exported constants must still be mapped:\n{code}"
        );
        for (generated_line, generated_column, source_line, source_column) in &tokens {
            let emitted = lines[*generated_line as usize];
            let original = sources[*source_line as usize];
            assert!(
                *generated_column as usize <= emitted.chars().count()
                    && *source_column as usize <= original.chars().count(),
                "gen {generated_line}:{generated_column} in {emitted:?} -> \
                 {source_line}:{source_column} in {original:?}"
            );
        }
        let second = lines
            .iter()
            .position(|line| line.starts_with("const SECOND"))
            .expect("the second constant is emitted without its `export`");
        let token = tokens
            .iter()
            .find(|(line, ..)| *line as usize == second)
            .expect("the second constant must be mapped");
        assert_eq!(
            sources[token.2 as usize], "export const SECOND = FIRST + 1;",
            "it must map to its own line"
        );
    }

    #[test]
    fn strips_typescript_and_lowers_modules() {
        let transformed = transform_module(
            Path::new("entry.ts"),
            r#"
                import value, { named as local } from "./dep.js";
                export const answer: number = local;
                export default function named() { return value + answer; }
            "#,
            Target::Server,
        );

        assert!(
            transformed.diagnostics.is_empty(),
            "{:?}",
            transformed.diagnostics
        );
        assert!(!transformed.code.contains(": number"));
        assert!(!transformed.code.contains("import value"));
        assert!(!transformed.code.contains("export const"));
        assert!(transformed.code.contains("require.esm(\"./dep.js\")"));
        assert!(transformed.code.contains("__export(exports,\"answer\""));
        assert!(transformed.code.contains("__export(exports,\"default\""));
    }

    /// The map is produced only when the build asks for one; nothing cheaper and
    /// guessed stands in for it when it does not.
    #[test]
    fn no_map_is_produced_when_the_build_did_not_ask_for_one() {
        let result = transform_module_in_language(
            Path::new("/app/src/a.ts"),
            "export const answer: number = 42\n",
            Target::Client,
            false,
            JsxExtensions::default(),
            &ProjectConfig::default(),
            SourceLanguage::FromPath,
            false,
        );
        assert!(
            result.map.is_none(),
            "no map was requested, so none is invented"
        );
    }

    /// Every token the printer produced must point at real text: a node the
    /// lowering synthesized carries a zero span, which would otherwise be printed
    /// as "line 1, column 0" of a file the code never came from.
    #[test]
    fn a_module_map_never_claims_the_first_byte_for_synthesized_code() {
        // The first line is a comment and the second an import: neither survives
        // into the lowered body, so nothing in the emitted code legitimately comes
        // from byte 0 — yet the lowering synthesizes `__import(ns, "dep")` nodes
        // for every use of the imported binding.
        let source = "// a comment nobody emits\nimport { dep } from './dep.js'\nexport const used = dep(globalThis.who)\n";
        let result = transform_module_in_language(
            Path::new("/app/src/a.ts"),
            source,
            Target::Client,
            false,
            JsxExtensions::default(),
            &ProjectConfig::default(),
            SourceLanguage::FromPath,
            true,
        );
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        let map = result.map.expect("a map was requested");
        assert!(
            !map.tokens().is_empty(),
            "the module must have real tokens: {}",
            result.code
        );
        assert!(
            map.tokens()
                .iter()
                .all(|token| (token.source_line, token.source_column) != (0, 0)),
            "no token may claim the file's first byte, got {:?}",
            map.tokens(),
        );
        // The rewritten import reference keeps the position of the identifier it
        // replaced, rather than collapsing to the start of the file.
        let dep_line = 2;
        assert!(
            map.tokens()
                .iter()
                .any(|token| token.source_line == dep_line),
            "the rewritten `dep` reference must still point at its own line, got {:?}",
            map.tokens(),
        );
    }

    #[test]
    fn lowers_literal_dynamic_import_into_the_single_chunk_runtime() {
        let transformed = transform_module(
            Path::new("entry.js"),
            "export const load = () => import('./lazy.js');",
            Target::Server,
        );
        assert!(
            transformed
                .code
                .contains("__dynamic(require, \"./lazy.js\")"),
            "{}",
            transformed.code
        );
    }

    /// FINDINGS #17. JSX whitespace is SIGNIFICANT between elements on the same line:
    /// `<b>x</b> <i>y</i>` keeps the single space as its own child, so a rendered UI
    /// does not lose the spaces its author wrote. The other two rules must hold at the
    /// same time: a whitespace run containing a newline is dropped entirely (so
    /// indentation never leaks into the page), and a multi-line text run joins its
    /// lines with exactly one space while interior runs of spaces are preserved
    /// verbatim.
    #[test]
    fn jsx_text_whitespace_follows_the_jsx_rules() {
        let transformed = transform_module(
            Path::new("component.jsx"),
            "export const C = () => (\n  <div>\n    <b>x</b> <i>y</i>\n    <p>\n      hello   world\n      again\n    </p>\n  </div>\n);\n",
            Target::Server,
        );
        assert!(
            transformed.diagnostics.is_empty(),
            "{:?}",
            transformed.diagnostics
        );
        let code = &transformed.code;
        // Same-line space between two elements survives as a child of its own.
        assert!(
            code.contains("\"b\", { children: \"x\" }),\n\t\" \",") || code.contains("\" \","),
            "the space between <b> and <i> must be preserved: {code}"
        );
        // Indentation-only children (they contain a newline) are dropped.
        assert!(
            !code.contains("\"\\n"),
            "indentation must not leak into children: {code}"
        );
        // Lines of one text run join with a single space; interior spaces are kept.
        assert!(code.contains("children: \"hello   world again\""), "{code}");
    }

    #[test]
    fn lowers_jsx_to_javascript() {
        let transformed = transform_module(
            Path::new("component.jsx"),
            "export const Component = ({ name }) => <div>Hello {name}</div>;",
            Target::Server,
        );
        assert!(
            transformed.diagnostics.is_empty(),
            "{:?}",
            transformed.diagnostics
        );
        assert!(!transformed.code.contains("<div>"));
        assert!(
            transformed
                .code
                .contains("require.esm(\"react/jsx-runtime\")")
        );
    }

    /// A `.vue` file's extension says nothing about the language of what its
    /// compiler emitted: `@vue/compiler-sfc` leaves a `<script lang="ts">`
    /// component's annotations in place for the bundler to strip (which is why
    /// a component compiler hands its own output to core as TypeScript). Parsing that
    /// by extension makes `(_ctx: any) => ...` a syntax error in generated code
    /// the app never wrote.
    #[test]
    fn compiled_component_source_is_parsed_in_the_caller_declared_language() {
        let compiled = "import { ref } from 'vue';\n\
                        const _sfc_main = { setup() { const n = ref<number>(0); \
                        return (_ctx: any, _cache: any) => n.value } };\n\
                        export default _sfc_main;\n";
        let typescript = transform_module_in_language(
            Path::new("/app/src/App.vue"),
            compiled,
            Target::Client,
            false,
            JsxExtensions::default(),
            &ProjectConfig::default(),
            SourceLanguage::TypeScript,
            false,
        );
        assert!(
            typescript.diagnostics.is_empty(),
            "{:?}",
            typescript.diagnostics
        );
        assert!(
            !typescript.code.contains("_ctx: any"),
            "{}",
            typescript.code
        );
        assert_eq!(typescript.dependencies, vec!["vue".to_string()]);

        // The same source read by extension (`.vue` is not TypeScript to oxc) is
        // a parse error — which is exactly the misreport this parameter exists to
        // prevent.
        let by_path = transform_module_in_language(
            Path::new("/app/src/App.vue"),
            compiled,
            Target::Client,
            false,
            JsxExtensions::default(),
            &ProjectConfig::default(),
            SourceLanguage::FromPath,
            false,
        );
        assert!(
            !by_path.diagnostics.is_empty(),
            "reading compiled TypeScript as JavaScript must not silently succeed"
        );
    }

    /// `jsxImportSource` names the package the automatic runtime imports from. A
    /// preact app has no `react` in `node_modules` at all, so lowering against
    /// `react/jsx-runtime` is not a cosmetic difference — it is an unresolvable
    /// import and a build that cannot emit.
    #[test]
    fn a_custom_import_source_replaces_react_jsx_runtime() {
        let transformed = transform_module_with_options(
            Path::new("/app/src/app.tsx"),
            "export const App = () => <div>hi</div>;",
            Target::Client,
            false,
            JsxExtensions::default(),
            &JsxConfig {
                import_source: Some("preact".to_string()),
                ..JsxConfig::default()
            },
        );
        assert!(
            transformed.diagnostics.is_empty(),
            "{:?}",
            transformed.diagnostics
        );
        assert_eq!(
            transformed.dependencies,
            vec!["preact/jsx-runtime".to_string()],
            "the runtime import IS the dependency the graph must resolve: {}",
            transformed.code
        );
    }

    /// The classic runtime's factory import must SURVIVE TypeScript's import
    /// elision. oxc decides whether `import { h } from 'preact'` is type-only by
    /// comparing the binding against `TypeScriptOptions::jsx_pragma` (default
    /// `React.createElement`); left at the default the import is dropped and the
    /// build "succeeds" into a bundle that dies with `h is not defined`.
    #[test]
    fn the_classic_factory_import_is_not_elided_as_a_type_import() {
        let transformed = transform_module_with_options(
            Path::new("/app/src/app.tsx"),
            "import { h, Fragment } from 'preact';\nexport const App = () => <><div>hi</div></>;",
            Target::Client,
            false,
            JsxExtensions::default(),
            &JsxConfig {
                runtime: Some(JsxRuntime::Classic),
                factory: Some("h".to_string()),
                fragment_factory: Some("Fragment".to_string()),
                ..JsxConfig::default()
            },
        );
        assert!(
            transformed.diagnostics.is_empty(),
            "{:?}",
            transformed.diagnostics
        );
        assert!(
            transformed.code.contains("\"h\"") && transformed.code.contains("\"Fragment\""),
            "the classic factory/fragment must be called: {}",
            transformed.code
        );
        assert_eq!(
            transformed.dependencies,
            vec!["preact".to_string()],
            "the factory's import must survive elision (and nothing else is imported): {}",
            transformed.code
        );
    }

    /// A file-level `@jsxImportSource` pragma outranks the project's configuration
    /// (oxc rescans the leading comments after our options are installed). Pinned
    /// because it is free today and an oxc bump could take it away unnoticed.
    ///
    /// The pragma alone would pass with the configured source never applied at all
    /// (oxc reads the comment itself), so the SAME configuration is asserted on a
    /// module without a pragma: this test fails both if the pragma stops winning
    /// and if [`JsxConfig::apply`] stops being called.
    #[test]
    fn a_jsx_import_source_pragma_beats_the_configured_source() {
        let configured = JsxConfig {
            import_source: Some("myjsx".to_string()),
            ..JsxConfig::default()
        };
        let with_pragma = transform_module_with_options(
            Path::new("/app/src/app.tsx"),
            "/** @jsxImportSource preact */\nexport const App = () => <div>hi</div>;",
            Target::Client,
            false,
            JsxExtensions::default(),
            &configured,
        );
        assert!(
            with_pragma.diagnostics.is_empty(),
            "{:?}",
            with_pragma.diagnostics
        );
        assert_eq!(
            with_pragma.dependencies,
            vec!["preact/jsx-runtime".to_string()],
            "the pragma must win over the configured source: {}",
            with_pragma.code
        );

        let without_pragma = transform_module_with_options(
            Path::new("/app/src/sibling.tsx"),
            "export const Sibling = () => <div>hi</div>;",
            Target::Client,
            false,
            JsxExtensions::default(),
            &configured,
        );
        assert_eq!(
            without_pragma.dependencies,
            vec!["myjsx/jsx-runtime".to_string()],
            "a module with no pragma must take the CONFIGURED source: {}",
            without_pragma.code
        );
    }

    /// oxc infers JSX purity only for a config that names neither an import source
    /// nor a pragma, so a hand-built `JsxOptions` (with `pure: false`) would drop
    /// every `/*#__PURE__*/` annotation the moment an import source is set — and
    /// silently de-tree-shake the bundle. Pins that the options are MUTATED.
    #[test]
    fn pure_annotations_survive_a_custom_import_source() {
        let source = "export const App = () => <div>hi</div>;";
        let default = transform_module(Path::new("/app/src/app.tsx"), source, Target::Client);
        let custom = transform_module_with_options(
            Path::new("/app/src/app.tsx"),
            source,
            Target::Client,
            false,
            JsxExtensions::default(),
            &JsxConfig {
                import_source: Some("preact".to_string()),
                ..JsxConfig::default()
            },
        );
        assert!(
            default.code.contains("@__PURE__"),
            "baseline must be annotated: {}",
            default.code
        );
        assert!(
            custom.code.contains("@__PURE__"),
            "a custom import source must not cost the pure annotations: {}",
            custom.code
        );
    }

    #[test]
    fn refresh_transform_injects_registrations_only_when_enabled() {
        let source = "export const Navbar = () => <nav>hi</nav>;";
        // Off by default: no Fast Refresh instrumentation (production is untouched).
        let plain = transform_module(Path::new("Navbar.jsx"), source, Target::Client);
        assert!(
            !plain.code.contains("$RefreshReg$") && !plain.code.contains("$RefreshSig$"),
            "refresh must be OFF by default: {}",
            plain.code
        );
        // On: oxc's native Fast Refresh transform registers the component so the
        // runtime can swap it in place — no Node/babel involved.
        let refreshed = transform_module_with_options(
            Path::new("Navbar.jsx"),
            source,
            Target::Client,
            true,
            JsxExtensions::default(),
            &JsxConfig::default(),
        );
        assert!(
            refreshed.code.contains("$RefreshReg$"),
            "refresh ON must register the component: {}",
            refreshed.code
        );
    }

    /// Permissive JSX mode must still yield dependencies, because a
    /// fatal parse returns a dummy program and the importer's whole subtree
    /// (`components/Gallery.js`, ...) silently vanishes from the graph.
    #[test]
    fn permissive_jsx_mode_compiles_js_and_keeps_its_imports() {
        let source = r#"
            import Gallery from "../components/Gallery";
            export default function Home() {
                return <Gallery title="hi" />;
            }
        "#;
        let transformed = transform_module_with_options(
            Path::new("pages/index.js"),
            source,
            Target::Client,
            false,
            JsxExtensions::JsxInJavaScript,
            &JsxConfig::default(),
        );

        assert!(
            transformed.diagnostics.is_empty(),
            "{:?}",
            transformed.diagnostics
        );
        assert!(
            transformed.code.contains("react/jsx-runtime"),
            "JSX must be lowered through the automatic runtime: {}",
            transformed.code
        );
        assert_eq!(
            transformed.dependencies,
            ["../components/Gallery", "react/jsx-runtime"]
        );
    }

    /// Explicit-extension mode rejects the same file with an actionable message.
    #[test]
    fn explicit_jsx_mode_rejects_js_with_an_actionable_error() {
        let transformed = transform_module_with_options(
            Path::new("src/main.js"),
            "export default function App() { return <div />; }",
            Target::Client,
            false,
            JsxExtensions::JsxAndTsxOnly,
            &JsxConfig::default(),
        );

        let fatal: Vec<&TransformDiagnostic> =
            transformed.diagnostics.iter().filter(|d| d.fatal).collect();
        assert_eq!(fatal.len(), 1, "{:?}", transformed.diagnostics);
        let message = &fatal[0].message;
        assert!(
            message.contains("JSX is not enabled for `.js` files"),
            "{message}"
        );
        assert!(message.contains("main.jsx"), "{message}");
        assert!(
            !message.contains("Unexpected JSX expression"),
            "the raw oxc message must be replaced: {message}"
        );
    }

    /// The rule is scoped, not a blanket "JSX everywhere": in TypeScript `<T>x` is a
    /// type assertion, so a `.ts` module stays JSX-free in permissive mode.
    #[test]
    fn permissive_jsx_mode_keeps_ts_angle_bracket_assertions() {
        let transformed = transform_module_with_options(
            Path::new("lib/cast.ts"),
            "const value: unknown = 1; export const text = <string>value;",
            Target::Server,
            false,
            JsxExtensions::JsxInJavaScript,
            &JsxConfig::default(),
        );

        assert!(
            transformed.diagnostics.is_empty(),
            "{:?}",
            transformed.diagnostics
        );
        assert!(
            !transformed.code.contains("jsx-runtime"),
            "a `.ts` type assertion must not become a JSX element: {}",
            transformed.code
        );
    }

    #[test]
    fn records_imported_symbol_demand_without_scanning_generated_code() {
        let transformed = transform_module(
            Path::new("entry.js"),
            r#"
                import { used, unused } from "./values.js";
                import "./effects.js";
                export const answer = used;
            "#,
            Target::Server,
        );

        let values = transformed
            .dependency_demands
            .iter()
            .find(|demand| demand.specifier == "./values.js")
            .unwrap();
        assert!(!values.all);
        assert_eq!(values.names, ["used"]);
        assert!(!values.dynamic);

        let effects = transformed
            .dependency_demands
            .iter()
            .find(|demand| demand.specifier == "./effects.js")
            .unwrap();
        assert!(!effects.all);
        assert!(effects.names.is_empty());
        assert!(!effects.dynamic);
    }
}
