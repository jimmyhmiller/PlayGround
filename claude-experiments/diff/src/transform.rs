use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use oxc_allocator::{Allocator, TakeIn};
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
use oxc_transformer::{JsxRuntime as OxcJsxRuntime, ReactRefreshOptions, TransformOptions, Transformer};

use crate::frontend_profile::{self, Phase};
use crate::source_map::{LineTrack, MapOrigin, MapToken, ModuleSourceMap};
use crate::parser::{JsxExtensions, collect_dependencies, collect_dynamic_dependencies};

/// One oxc parse/semantic/transform diagnostic, with its severity preserved.
/// An error means the code oxc produced does not match the source; a warning
/// leaves runnable code, so the two cannot be collapsed into one string list —
/// the bundler decides fatality from `fatal`.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TransformDiagnostic {
    pub fatal: bool,
    pub message: String,
}

impl TransformDiagnostic {
    /// A diffpack-produced transform failure. The module's code is empty when one
    /// of these is returned, so it is always fatal.
    pub fn error(message: String) -> Self {
        Self {
            fatal: true,
            message,
        }
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

/// Replaces oxc's bare `Unexpected JSX expression` with an actionable message when
/// the module was parsed WITHOUT JSX. The raw diagnostic points at a column and
/// says nothing about why the same source compiles under Next and not here, which
/// sent a real investigation down the wrong path.
///
/// `esbuild.include`/`esbuild.loader` is deliberately NOT offered as a remedy:
/// diffpack does not read it (`ResolvedViteConfig` has no `esbuild` field), and
/// pointing at a knob that does nothing is a silent fallback wearing a help string.
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
    let stem = path.file_stem().and_then(|stem| stem.to_str()).unwrap_or("module");
    let explanation = match extension {
        "js" | "mjs" | "cjs" => format!(
            "JSX is not enabled for `.{extension}` files. Vite/esbuild parse `.{extension}` as \
             plain JavaScript on purpose and diffpack matches that; \
             `esbuild.include`/`esbuild.loader` is not honored. Rename it to `{stem}.jsx`. \
             (A Next.js project, which does allow JSX in `.js`, is detected automatically.)"
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

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TransformResult {
    pub code: String,
    pub diagnostics: Vec<TransformDiagnostic>,
    pub is_esm: bool,
    pub dependencies: Vec<String>,
    pub dependency_demands: Vec<DependencyDemand>,
    pub flat_module: Option<FlatModule>,
    pub liveness: ModuleLiveness,
    /// The module `await`s at its top level. Only representable in ESM output
    /// where the module's statements stay at the top level of the chunk; the
    /// emit refuses (a hard, module-naming error) rather than rendering an
    /// `await` into a synchronous factory or CommonJS wrapper.
    pub uses_top_level_await: bool,
    /// The module references `import.meta` (beyond an opted-in
    /// `import.meta.env`, which is rewritten before the transform). Valid in
    /// ESM output (where it resolves against the emitted chunk, the standard
    /// bundler semantic); a syntax error in CommonJS output, so the emit
    /// refuses there.
    pub uses_import_meta: bool,
    /// The module freely references a CommonJS ambient (`exports`, `module`,
    /// `require`, `__filename`, `__dirname`). Such a module needs the factory
    /// wrapper that defines them and must not be scope-hoisted into ESM output.
    pub uses_cjs_globals: bool,
    /// The module freely references `__dirname` or `__filename` specifically.
    /// A Node target resolves those from the emitted bundle's own location, but
    /// a BROWSER target has no such thing: the emit must define them per module
    /// or the reference is a `ReferenceError` at module init. Strictly a subset
    /// of [`Self::uses_cjs_globals`].
    pub uses_dirname: bool,
    /// Module-worker entries this module creates via
    /// `new Worker(new URL('<specifier>', import.meta.url))`, as
    /// `(placeholder_key, specifier)`. The placeholder (already substituted
    /// into the code) is `__diffpack_worker__<key>__`; the bundler resolves
    /// the specifier and the emit replaces the placeholder with the emitted
    /// worker bundle's public URL.
    pub workers: Vec<(String, String)>,
    /// The REAL source map over [`Self::code`], as the Oxc printer emitted it:
    /// every token is a position the printer actually wrote, paired with the span
    /// of the AST node it printed. `None` when the build did not ask for source
    /// maps (the map costs a second print per module), and `None` for a module
    /// whose code was not printed from an AST at all — never a guessed map. See
    /// [`crate::source_map`].
    pub map: Option<ModuleSourceMap>,
}

/// The export/import structure of a module, at the granularity the generic
/// dead-module elimination pass ([`crate::bundler`]) needs to compute
/// export-level liveness across the graph.
///
/// The distinction that makes barrel tree-shaking possible is between a **body
/// use** (an imported binding referenced in real module code — a demand that
/// applies unconditionally once the module runs) and a **re-export** (an
/// imported binding merely forwarded as one of this module's own exports — a
/// demand that applies only if that export is itself used). A module reached
/// only through a barrel whose re-exported binding no live module uses places no
/// body-use demand on its source, so a `sideEffects:false` source becomes
/// droppable.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct ModuleLiveness {
    /// Every explicit export name of this module (locally-defined exports,
    /// `default`, named re-exports, and `export * as ns`). Bare `export *` adds
    /// no name here — it is tracked in [`Self::star_reexports`].
    pub exports: Vec<String>,
    /// Specifiers of bare `export * from S` — this module re-exports all of S's
    /// names.
    pub star_reexports: Vec<String>,
    /// Re-export edges: this module's export `exported` forwards the target's
    /// `imported` binding. `imported == "*"` is a namespace re-export
    /// (`export * as ns from S`).
    pub reexports: Vec<ReExport>,
    /// Genuine body-level demand per dependency specifier (names referenced in
    /// real code, plus `all` for a namespace binding used in the body). Applies
    /// unconditionally once this module is live.
    pub body_uses: Vec<BodyUse>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ReExport {
    pub specifier: String,
    pub imported: String,
    pub exported: String,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BodyUse {
    pub specifier: String,
    pub all: bool,
    pub names: Vec<String>,
}

#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct DependencyDemand {
    pub specifier: String,
    pub all: bool,
    pub names: Vec<String>,
    pub dynamic: bool,
    /// Every reference to this specifier is a `require(...)` inside a `try` block —
    /// see [`crate::parser::collect_optional_dependencies`]. A resolution failure is
    /// then the program's own designed path, not a broken build.
    pub optional: bool,
    /// At least one reference to this specifier is a CommonJS `require(...)` call,
    /// so it must resolve under the `require` export condition. See
    /// [`crate::parser::DependencySyntax`].
    pub require_syntax: bool,
    /// At least one reference to this specifier is an ESM form (a static
    /// `import` / `export … from`, or a dynamic `import()`), so it must resolve
    /// under the `import` export condition.
    pub import_syntax: bool,
    /// At least one reference to this specifier needs the target ALREADY EVALUATED
    /// when this module's body runs: a static `import`, an `export … from`, or a
    /// CommonJS `require(...)`. See [`crate::parser::collect_eager_dependencies`].
    pub eager: bool,
}

impl DependencyDemand {
    /// Whether this edge may be DEFERRED — i.e. whether the target is allowed to live
    /// in a chunk that is only fetched when the `import()` runs.
    ///
    /// `dynamic` alone does not answer that. `dynamic` means "there is an `import()`
    /// call site here to lower"; `eager` means "there is also a reference that reads the
    /// target synchronously". A module reached BOTH ways — the barrel that re-exports a
    /// component and also `dynamic(() => import(...))`s it — is not a chunk boundary at
    /// all: its static reference resolves against the registry the instant the barrel
    /// evaluates, long before any chunk fetch could have completed. Splitting it out
    /// produced `Module is not loaded: <id>` at first render.
    ///
    /// Every graph question about REACHABILITY (static closure, execution order, chunk
    /// membership and prerequisites) must ask this; only the questions about the
    /// `import()` CALL SITE (lowering it, and giving it a chunk to fetch) ask `dynamic`.
    pub fn deferred(&self) -> bool {
        self.dynamic && !self.eager
    }
}

/// Which runtime a module's JSX is lowered with.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum JsxRuntime {
    /// `import { jsx as _jsx } from "<import source>/jsx-runtime"` — TypeScript's
    /// `"jsx": "react-jsx"`, and the default for every project that says nothing.
    #[default]
    Automatic,
    /// `React.createElement(...)` (or whatever `jsxFactory` names) — TypeScript's
    /// `"jsx": "react"`.
    Classic,
}

/// How a module's JSX is lowered: the resolved `jsx` / `jsxImportSource` /
/// `jsxFactory` / `jsxFragmentFactory` contract for one file.
///
/// Every field is optional, and `Default` (all `None`) means the automatic runtime
/// against `react` — oxc's own default, so a project that configures nothing is
/// byte-identical to before this type existed. The layers that fill it in, weakest
/// first, are: the tsconfig that owns the file, then the build's own settings
/// (`vite.config`'s `esbuild.*` / `oxc.jsx`), then a per-file `@jsxImportSource` /
/// `@jsx` / `@jsxFrag` / `@jsxRuntime` pragma — the last of which oxc applies
/// itself, from the program's leading comments, after these options are installed.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct JsxConfig {
    pub runtime: Option<JsxRuntime>,
    /// The package the automatic runtime imports from (`preact` -> `preact/jsx-runtime`).
    pub import_source: Option<String>,
    /// The classic-runtime factory (`h`, `React.createElement`).
    pub factory: Option<String>,
    /// The classic-runtime fragment factory (`Fragment`, `React.Fragment`).
    pub fragment_factory: Option<String>,
}

impl JsxConfig {
    /// This config with every field `other` sets replaced by `other`'s. Field-by-
    /// field, not whole-record, because that is exactly how Vite layers its own
    /// options over the tsconfig's: it nulls only the tsconfig fields whose
    /// counterpart the config sets.
    #[must_use]
    pub fn overridden_by(mut self, other: &Self) -> Self {
        if other.runtime.is_some() {
            self.runtime = other.runtime;
        }
        if other.import_source.is_some() {
            self.import_source.clone_from(&other.import_source);
        }
        if other.factory.is_some() {
            self.factory.clone_from(&other.factory);
        }
        if other.fragment_factory.is_some() {
            self.fragment_factory.clone_from(&other.fragment_factory);
        }
        self
    }

    /// Installs this config on oxc's transform options.
    ///
    /// MUTATES the caller's defaults rather than constructing a fresh `JsxOptions`:
    /// `JsxOptions::enable()` sets `pure: true`, and oxc only infers purity for a
    /// config that names neither an import source nor a pragma
    /// (`jsx_impl.rs`: `pure || (import_source.is_none() && pragma.is_none())`), so a
    /// hand-built `JsxOptions` that sets `import_source` would silently drop every
    /// `/*#__PURE__*/` annotation and de-tree-shake the bundle.
    fn apply(&self, options: &mut TransformOptions) {
        match self.runtime.unwrap_or_default() {
            JsxRuntime::Automatic => {
                options.jsx.runtime = OxcJsxRuntime::Automatic;
                if let Some(import_source) = &self.import_source {
                    options.jsx.import_source = Some(import_source.clone());
                }
            }
            JsxRuntime::Classic => {
                options.jsx.runtime = OxcJsxRuntime::Classic;
                if let Some(factory) = &self.factory {
                    options.jsx.pragma = Some(factory.clone());
                    // In lockstep, and load-bearing: oxc's TypeScript pass decides
                    // whether `import { h } from 'preact'` is a type-only import by
                    // comparing the binding against `TypeScriptOptions::jsx_pragma`,
                    // which defaults to `React.createElement`. Left at the default,
                    // the factory's import is elided and the build "succeeds" into a
                    // bundle that dies with `h is not defined`.
                    options.typescript.jsx_pragma = factory.clone().into();
                }
                if let Some(fragment_factory) = &self.fragment_factory {
                    options.jsx.pragma_frag = Some(fragment_factory.clone());
                    options.typescript.jsx_pragma_frag = fragment_factory.clone().into();
                }
            }
        }
    }
}

/// How a module's `@decorator`s are lowered: the resolved `experimentalDecorators` /
/// `emitDecoratorMetadata` / `strictNullChecks` contract for one file, read off the
/// tsconfig that owns it.
///
/// A decorator is SYNTAX no JavaScript engine accepts — Node and every browser
/// reject `@dec class C {}` at parse time — so unlike JSX there is no "leave it to
/// the runtime" option: either the emit lowers it or the emit is unloadable. Which
/// lowering is correct is not a bundler preference either. TypeScript's legacy
/// decorators (`experimentalDecorators: true`) and the TC39 Stage 3 decorators call
/// the decorator with different arguments and assign its return value differently,
/// so a file's own tsconfig is the only thing that can say which semantics its
/// authors wrote against.
///
/// `Default` (all false) is TypeScript's own default for a project that says
/// nothing: Stage 3 semantics.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct DecoratorConfig {
    /// tsconfig `experimentalDecorators`: TypeScript's pre-standard decorators.
    pub legacy: bool,
    /// tsconfig `emitDecoratorMetadata`. TypeScript ignores it unless
    /// `experimentalDecorators` is on, and so does the transform.
    pub emit_metadata: bool,
    /// tsconfig `strictNullChecks` (which `strict` turns on). Only read when
    /// metadata is emitted: it decides whether `T | null` records `T`'s constructor
    /// or `Object` in a `design:type`.
    pub strict_null_checks: bool,
}

impl DecoratorConfig {
    /// Installs this config on oxc's transform options.
    fn apply(&self, options: &mut TransformOptions) {
        options.decorator.legacy = self.legacy;
        options.decorator.emit_decorator_metadata = self.emit_metadata;
        options.decorator.strict_null_checks = self.strict_null_checks;
    }
}

/// The compilation contract one module inherits from the tsconfig/jsconfig that
/// owns it (layered, for JSX, with the build's own settings). Both halves answer
/// the same question — how must THIS file's syntax be lowered — and both are read
/// from the same config, so they travel together rather than as separate arguments
/// that a caller could resolve from two different configs.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct ProjectConfig {
    pub jsx: JsxConfig,
    pub decorators: DecoratorConfig,
}

/// The environment a module is being compiled for. TanStack Start ships
/// environment-neutral runtime stubs (`createServerOnlyFn`, `createClientOnlyFn`,
/// `createIsomorphicFn`) and relies on the build tool to specialize them per
/// environment. On the client this specialization is what lets whole-program
/// tree-shaking drop server-only code (see [`apply_env_transform`]); the
/// `Server` build keeps the neutral runtime stubs (which already behave
/// correctly under Node) and so applies no transform — it is the default.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum Target {
    /// The browser build. Server-only functions are replaced with throwing
    /// stubs and isomorphic functions collapse to their client implementation,
    /// severing the references that would otherwise pull server modules
    /// (e.g. `node:async_hooks`) into the client graph.
    Client,
    /// A server build (`ssr`/`nitro`). No transform: the neutral runtime stubs
    /// resolve to the correct behavior under Node.
    #[default]
    Server,
    /// The React Server (RSC) build. Resolves under the `react-server` export
    /// condition (a different React than SSR/client), and specializes the RSC
    /// module boundaries: a `"use client"` module becomes client-reference
    /// re-exports (`createClientModuleProxy`, so no component code enters this
    /// graph), and a `"use server"` module becomes server references. Otherwise
    /// server-like — the TanStack env helpers and node resolution behave as on
    /// `Server`.
    ReactServer,
}

impl Target {
    /// The graph's name, for anything that has to tell one graph's build products
    /// from another's (a source-map `sources` label, which is a module's identity).
    pub fn label(self) -> &'static str {
        match self {
            Self::Client => "client",
            Self::Server => "server",
            Self::ReactServer => "react-server",
        }
    }
}

#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct FlatModule {
    pub code: String,
    /// Which line of the module's LOWERED code each line of [`Self::code`] came
    /// from, so the module's real source map survives the flat derivation. `None`
    /// when the build did not ask for source maps.
    pub map_lines: Option<LineTrack>,
    pub declarations: Vec<String>,
    pub exports: Vec<String>,
    pub has_direct_effects: bool,
    pub import_replacements: Vec<(String, String)>,
    pub foldable: Option<FoldableModule>,
}

#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct FoldableModule {
    pub constants: Vec<(String, FoldExpression)>,
    pub console_logs: Vec<FoldExpression>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum FoldExpression {
    Number(u64),
    Reference(String),
    Add(Box<Self>, Box<Self>),
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

/// Transform a single TS/TSX/JS/JSX module to STANDALONE, runnable ES module source:
/// strip TypeScript, lower JSX via the automatic runtime (`react/jsx-runtime`), and
/// leave every `import`/`export` specifier UNTOUCHED (no diffpack bundler rewriting).
/// The result runs directly under Node from the module's own directory, resolving its
/// imports (`react/jsx-runtime`, `@vercel/og`/`next/og`, ...) through Node's resolver.
/// Used by the build-time `@vercel/og` ImageResponse prerender. Returns the emitted code
/// or a hard error carrying any parse/transform diagnostics (never silent).
pub fn transform_to_standalone_esm(path: &Path, source: &str) -> Result<String, String> {
    let allocator = Allocator::default();
    // The only caller is the Next app-router `@vercel/og` prerender, so Next's JSX
    // rule applies: an `ImageResponse` generator in a `.js` route file is JSX.
    let source_type = crate::parser::source_type_for(path, JsxExtensions::NextJs);
    let parsed = Parser::new(&allocator, source, source_type).parse();
    let mut diagnostics: Vec<String> =
        parsed.diagnostics.into_iter().map(|d| d.to_string()).collect();
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
/// `jsx` is the project's JSX-extension rule (see [`JsxExtensions`]): Next compiles
/// JSX in `.js`, Vite does not, and the same source must therefore parse
/// differently per project kind. `jsx_config` is how this module's JSX is LOWERED
/// (see [`JsxConfig`]) — the tsconfig/vite-config contract for the file, already
/// resolved by the caller.
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

/// Which language `source` is written in, when the module's own path cannot say.
///
/// A path answers this for every file diffpack reads off disk, but not for source
/// a compiler produced: `App.vue` compiled by `@vue/compiler-sfc` is TypeScript
/// whenever the SFC's `<script>` was, and `.vue` names neither. This is the
/// caller's explicit answer for those, and is exactly the choice
/// `@vitejs/plugin-vue` makes when it hands its own output to Vite's transform
/// with `lang: "ts"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SourceLanguage {
    /// Derive it from the extension — every module read straight off disk.
    #[default]
    FromPath,
    /// Plain JavaScript (no type annotations, no JSX).
    JavaScript,
    /// TypeScript without JSX. Component compilers emit `_ctx: any` style
    /// annotations, never JSX, so enabling JSX here would only make `a < b`
    /// ambiguous.
    TypeScript,
}

impl SourceLanguage {
    /// The oxc source type for a module at `path`. `FromPath` defers to the
    /// project's JSX-extension rule; the explicit variants override it.
    fn source_type(self, path: &Path, jsx: JsxExtensions) -> SourceType {
        match self {
            SourceLanguage::FromPath => crate::parser::source_type_for(path, jsx),
            SourceLanguage::JavaScript => SourceType::default().with_module(true),
            SourceLanguage::TypeScript => {
                SourceType::default().with_typescript(true).with_module(true)
            }
        }
    }
}

/// [`transform_module_with_options`] for source a component compiler produced:
/// the path is the component (`App.vue`), but the language is `language`, not
/// whatever the extension implies. See [`crate::sfc`].
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

    // MDX/Markdown (`.mdx`/`.md`): compile to JSX first (a native source-to-source
    // transform), then run the rest of the pipeline on the emitted JSX exactly as for a
    // `.tsx` page. A compile error becomes this module's diagnostic (empty code), never
    // a silent pass.
    let mdx_compiled = if crate::mdx::is_mdx_path(path) {
        match crate::mdx::compile(path, source) {
            Ok(compiled) => Some(compiled.jsx),
            Err(diagnostic) => {
                return TransformResult {
                    code: String::new(),
                    diagnostics: vec![TransformDiagnostic::error(diagnostic)],
                    is_esm: true,
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
        }
    } else {
        None
    };
    let source = mdx_compiled.as_deref().unwrap_or(source);

    // The React Server (RSC) graph specializes the module boundaries BEFORE any
    // other rewrite: a `"use client"` module is replaced by its client-reference
    // re-exports (so none of the component code reaches this graph, and the
    // rewritten source's `react-server-dom-webpack/server` import is collected as
    // a real dep by the normal parse below), and a `"use server"` module becomes
    // server references. The use-client rewrite must run before `route_split` — a
    // use-client route module collapses to a single client reference, not a split
    // module. Gated on `Target::ReactServer` so no other build pays for it.
    let rsc_override = if target == Target::ReactServer {
        match crate::rsc::detect_directive(path, source) {
            Some(crate::rsc::RscDirective::Client) => {
                match crate::rsc::transform_use_client_server(path, source) {
                    Ok(rewritten) => rewritten,
                    Err(error) => {
                        return TransformResult {
                            code: String::new(),
                            diagnostics: vec![TransformDiagnostic::error(error)],
                            is_esm: true,
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
                }
            }
            Some(crate::rsc::RscDirective::Server) => {
                match crate::rsc::transform_use_server_server(path, source) {
                    Ok(rewritten) => rewritten,
                    Err(error) => {
                        return TransformResult {
                            code: String::new(),
                            diagnostics: vec![TransformDiagnostic::error(error)],
                            is_esm: true,
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
                }
            }
            Some(crate::rsc::RscDirective::Cache) => {
                // `"use cache"` wraps every export of the module in a cache boundary:
                // each export is memoized (keyed by its arguments) and run inside a
                // `cacheTag()`/`cacheLife()` collection scope, and the collected tags are
                // recorded on the current request so the page is bustable by
                // `revalidateTag` (native reimplementation on diffpack's next/cache tag
                // registry + prerender-cache invalidation). An unwrappable construct
                // (`export ... from` / `export *`) is a hard, specific error, never a
                // silent pass-through that would drop the caching semantics.
                match crate::rsc::transform_use_cache_server(path, source) {
                    Ok(rewritten) => rewritten,
                    Err(error) => {
                        return TransformResult {
                            code: String::new(),
                            diagnostics: vec![TransformDiagnostic::error(error)],
                            is_esm: true,
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
                }
            }
            None => None,
        }
    } else {
        None
    };
    let source = rsc_override.as_deref().unwrap_or(source);

    // A route file's heavy properties are split into virtual `?tsr-split`
    // modules and replaced with lazy imports before the module is lowered; this
    // is what turns each route's component into its own code-split chunk. Non-
    // route modules return `None` cheaply and take the source unchanged.
    let split = crate::route_split::split_reference_route(path, source);
    let source = split.as_deref().unwrap_or(source);

    // `next/font/google` / `next/font/local` are build-time macros: rewrite each
    // `Geist({...})` call into the static `{ className, variable, style }` object
    // and drop the throwing import (the companion CSS is generated by the
    // app-router adapter). Gated on a cheap string check; non-font modules pay
    // nothing. Runs first so downstream transforms see plain objects.
    let next_font = match crate::next_font::transform_next_font(path, source) {
        Ok(rewritten) => rewritten,
        Err(error) => {
            return TransformResult {
                code: String::new(),
                diagnostics: vec![TransformDiagnostic::error(error)],
                is_esm: true,
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
    };
    let source = next_font.as_deref().unwrap_or(source);

    // A module defining `createServerFn(...).handler(fn)` is rewritten per target:
    // the client gets a thin RPC stub keyed by the function's deterministic id
    // (dropping the server handler body), the server keeps the real handler and
    // wraps an in-process runner. Gated on a cheap string check, so non-server-fn
    // modules pay nothing; an unsupported server-fn shape is a hard error, never a
    // silent miscompile.
    let server_fn = match crate::server_fn::transform_server_fns(path, source, target) {
        Ok(rewritten) => rewritten,
        Err(error) => {
            return TransformResult {
                code: String::new(),
                diagnostics: vec![TransformDiagnostic::error(error)],
                is_esm: true,
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
    };
    let source = server_fn.as_deref().unwrap_or(source);

    // The generic RSC `"use server"` module boundary (distinct from TanStack's
    // `createServerFn`): every export of a directive-marked module becomes a server
    // reference. Skipped when the `Target::ReactServer` rsc_override above already
    // handled this module (so it is never transformed twice). Gated on the cheap
    // `"use server"` substring, then confirmed by the AST prologue check.
    //
    // ONLY the react-server graph keeps the real bodies. `registerServerReference` lives
    // in the flight WRITER (`react-server-dom-webpack/server`), and that package's
    // `exports` map only names a real file under the `react-server` condition — every
    // other condition resolves to a module whose entire body is `throw new Error("The
    // React Server Writer cannot be used outside a react-server environment")`. So a
    // `"use server"` module built for ANY other graph must become a reference STUB.
    //
    // That is not a workaround, it is the layering: in the browser AND in the SSR graph
    // (which renders CLIENT components on the server) an action is never invoked
    // in-process. It is passed as a prop or handed to `<form action={fn}>`, where React
    // reads its `$$id`/`$$FORM_ACTION` — exactly what `createServerReference` provides.
    // The real handler is dispatched by the action resolver, which lives in the
    // react-server graph alone.
    let use_server = if rsc_override.is_none()
        && source.contains("use server")
        && crate::rsc::detect_directive(path, source) == Some(crate::rsc::RscDirective::Server)
    {
        match target {
            Target::Client | Target::Server => {
                crate::rsc::transform_use_server_client(path, source)
            }
            Target::ReactServer => {
                match crate::rsc::transform_use_server_server(path, source) {
                    Ok(rewritten) => rewritten,
                    Err(error) => {
                        return TransformResult {
                            code: String::new(),
                            diagnostics: vec![TransformDiagnostic::error(error)],
                            is_esm: true,
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
                }
            }
        }
    } else {
        None
    };
    let source = use_server.as_deref().unwrap_or(source);

    let transform_started = frontend_profile::start();
    let allocator = Allocator::default();
    // MDX compiled to JSX: parse as TSX (`.mdx`/`.md` are not recognized by from_path).
    let source_type = if mdx_compiled.is_some() {
        SourceType::default().with_typescript(true).with_jsx(true).with_module(true)
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
    // default names `@oxc-project/runtime` — a package no Next or Vite app installs,
    // so a decorator would lower into an unresolvable import. diffpack serves them
    // from its own binary instead; see [`crate::runtime_helpers`].
    transform_options.helper_loader.module_name =
        std::borrow::Cow::Borrowed(crate::runtime_helpers::HELPER_PACKAGE);
    // How this file's JSX is lowered. A file-level `@jsxImportSource`/`@jsx`/
    // `@jsxFrag`/`@jsxRuntime` pragma still wins: oxc rescans the program's leading
    // comments and overrides these options before building the JSX pass.
    project_config.jsx.apply(&mut transform_options);
    // How this file's `@decorator`s are lowered, from the tsconfig that owns it.
    project_config.decorators.apply(&mut transform_options);
    if !project_config.decorators.legacy && let Some(name) = first_decorator_name(&program) {
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

    // Specialize environment-neutral TanStack Start runtime stubs for the target
    // BEFORE demand is computed. On the client this severs the references from
    // isomorphic/server-only wrappers to their server implementations, so the
    // now-unused server imports (e.g. `@tanstack/start-storage-context`, which
    // pulls `node:async_hooks`) are pruned by the existing side-effect-free
    // tree-shaking instead of leaking into the browser bundle. Because the
    // transform deletes references, scoping must be rebuilt so the demand pass
    // sees the imports as unreferenced.
    let mut scoping = transformed.scoping;
    if apply_env_transform(&allocator, &mut program, &scoping, target, path) {
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

    // WHICH TEXT the module map's positions refer to. Every stage above that
    // rewrites the source before it is parsed makes the printer's spans index
    // the REWRITTEN text, so a map that claimed the file on disk would point a
    // debugger at bytes those positions were never measured against. The map is
    // labelled with the stage instead, and carries that exact text as its
    // `sourcesContent` — see [`MapOrigin`].
    let map_origin = if use_server.is_some() {
        MapOrigin::Generated("use-server")
    } else if server_fn.is_some() {
        MapOrigin::Generated("server-fn")
    } else if next_font.is_some() {
        MapOrigin::Generated("next-font")
    } else if split.is_some() {
        MapOrigin::Generated("route-split")
    } else if rsc_override.is_some() {
        MapOrigin::Generated("rsc-directive")
    } else if mdx_compiled.is_some() {
        MapOrigin::Generated("mdx")
    } else {
        MapOrigin::File
    };
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
                    named_imports
                        .insert(local, (specifier.clone(), import.imported.name().to_string()));
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
    // cal.com's `const { i18n } = require("@calcom/i18n/next-i18next.config")` became
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

    fn visit_identifier_reference(
        &mut self,
        identifier: &oxc_ast::ast::IdentifierReference<'a>,
    ) {
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
    // module — route splitting injects a second `import { lazyFn } from
    // '@tanstack/react-router'` beside the original `import { createFileRoute }`.
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
    // "react"` (next-i18next's `appWithTranslation.js` ships it) — and it failed with
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
                    Some(exported) => export_getter(
                        &exported.name(),
                        &format!("require.esm({request})"),
                    ),
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
            match build_module_map(allocator, program, request, &reference_fragments, &placements, &code) {
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
    let guard = Statement::new_empty_statement(
        oxc_span::Span::new(end_of_source, end_of_source),
        &builder,
    );
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
        && reference_fragments
            .iter()
            .all(|fragment| matches(fragment))
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
        let alignment =
            align_fragment(reference, placement.dropped, aligned).ok_or_else(|| {
                format!(
                    "statement {} does not begin with the `{}` the lowering dropped, so the \
                     printer's positions cannot be placed in it",
                    placement.statement,
                    placement.dropped.trim_end()
                )
            })?;
        let (skip_line, skip_column) = alignment.skip;
        let first = printed_tokens
            .partition_point(|token| (token.get_dst_line(), token.get_dst_col()) < (start_line, start_column));
        let last = printed_tokens
            .partition_point(|token| (token.get_dst_line(), token.get_dst_col()) < (end_line, end_column));
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
            let (generated_line, generated_column) = if (local_line, local_column)
                < (skip_line, skip_column)
            {
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
        Expression::TemplateLiteral(template) => {
            template.expressions.iter().all(expression_is_obviously_pure)
        }
        Expression::ArrayExpression(array) => {
            array.elements.iter().all(|element| match element {
                oxc_ast::ast::ArrayExpressionElement::Elision(_) => true,
                oxc_ast::ast::ArrayExpressionElement::SpreadElement(_) => false,
                element => element
                    .as_expression()
                    .is_some_and(expression_is_obviously_pure),
            })
        }
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

/// Which TanStack Start environment-directive helper an imported binding refers
/// to. These are `@tanstack/*` runtime stubs that a build tool is expected to
/// specialize per environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnvFn {
    ServerOnly,
    ClientOnly,
    Isomorphic,
    Middleware,
}

/// Specializes TanStack Start's environment-directive helpers for `target`,
/// mirroring `@tanstack/start-plugin-core`'s `handleEnvOnly` /
/// `handleCreateIsomorphicFn` compiler passes:
///
/// - `createServerOnlyFn(fn)` keeps `fn` on the server; on the client it becomes
///   a throwing stub (the reference to `fn` is dropped).
/// - `createClientOnlyFn(fn)` is the mirror image.
/// - `createIsomorphicFn().client(a).server(b)` collapses to `a` on the client
///   and `b` on the server (or `() => {}` when the chosen environment has no
///   implementation).
/// - `createMiddleware()...server(fn)` drops its `.server`/`.validator`/
///   `.inputValidator` calls on the client, severing references to server-only
///   code (e.g. an API route's `getRequestHeaders`).
///
/// Only helpers imported from a `@tanstack/` package are matched, resolved by
/// symbol so a same-named local binding is never rewritten. Returns whether any
/// rewrite happened, so the caller can rebuild scoping (the pass deletes
/// references, which the demand computation must observe to prune the
/// now-unused server imports). This is currently a no-op for `Target::Server`,
/// whose neutral runtime stubs already behave correctly under Node.
fn apply_env_transform<'a>(
    allocator: &'a Allocator,
    program: &mut Program<'a>,
    scoping: &Scoping,
    target: Target,
    path: &Path,
) -> bool {
    if target != Target::Client {
        return false;
    }
    // A `@tanstack/*` package bundles these environment-directive helpers as
    // *local* modules (`createServerOnlyFn` from `./envOnly.js`,
    // `createIsomorphicFn` from `./createIsomorphicFn.js`), and its own modules
    // import them by relative specifier rather than through the package name. The
    // reference TanStack plugin matches these helpers by their well-known names
    // regardless of import source; mirror that, but only inside a `@tanstack`
    // package so a same-named helper in the user's own app is never rewritten.
    let in_tanstack_package = path
        .components()
        .any(|component| component.as_os_str() == "@tanstack");
    let mut kinds: HashMap<SymbolId, EnvFn> = HashMap::new();
    for statement in &program.body {
        let Statement::ImportDeclaration(declaration) = statement else {
            continue;
        };
        let specifier = declaration.source.value.as_str();
        let is_directive_source = specifier.starts_with("@tanstack/")
            || (in_tanstack_package && (specifier.starts_with("./") || specifier.starts_with("../")));
        if !is_directive_source {
            continue;
        }
        let Some(specifiers) = &declaration.specifiers else {
            continue;
        };
        for specifier in specifiers {
            let ImportDeclarationSpecifier::ImportSpecifier(specifier) = specifier else {
                continue;
            };
            let kind = match specifier.imported.name().as_str() {
                "createServerOnlyFn" => EnvFn::ServerOnly,
                "createClientOnlyFn" => EnvFn::ClientOnly,
                "createIsomorphicFn" => EnvFn::Isomorphic,
                "createMiddleware" => EnvFn::Middleware,
                _ => continue,
            };
            kinds.insert(specifier.local.symbol_id(), kind);
        }
    }
    if kinds.is_empty() {
        return false;
    }
    let mut transform = EnvTransform {
        allocator,
        scoping,
        kinds,
        target,
        changed: false,
    };
    transform.visit_program(program);
    transform.changed
}

struct EnvTransform<'a, 's> {
    allocator: &'a Allocator,
    scoping: &'s Scoping,
    kinds: HashMap<SymbolId, EnvFn>,
    target: Target,
    changed: bool,
}

impl<'a> EnvTransform<'a, '_> {
    /// The [`EnvFn`] an identifier reference resolves to, if it is one of the
    /// tracked `@tanstack/*` imports.
    fn env_fn(&self, identifier: &oxc_ast::ast::IdentifierReference<'a>) -> Option<EnvFn> {
        let reference_id = identifier.reference_id.get()?;
        let symbol_id = self.scoping.get_reference(reference_id).symbol_id()?;
        self.kinds.get(&symbol_id).copied()
    }

    /// Parses a constant JavaScript expression into this module's arena. Used to
    /// synthesize the throwing / empty-arrow replacements.
    fn parse_expression(&self, source: &'static str) -> Expression<'a> {
        let parsed = Parser::new(self.allocator, source, SourceType::default()).parse();
        let mut program = parsed.program;
        match program.body.first_mut() {
            Some(Statement::ExpressionStatement(statement)) => {
                statement.expression.take_in(&self.allocator)
            }
            _ => unreachable!("env-transform replacement source must be a single expression"),
        }
    }

    fn throwing_stub(&self, function: &str, environment: &str) -> Expression<'a> {
        // A distinct constant per (function, environment) so the parser sees a
        // 'static string; the set is closed and tiny.
        let source = match (function, environment) {
            ("createServerOnlyFn", "server") => {
                "(() => { throw new Error(\"createServerOnlyFn() functions can only be called on the server!\") })"
            }
            ("createClientOnlyFn", "client") => {
                "(() => { throw new Error(\"createClientOnlyFn() functions can only be called on the client!\") })"
            }
            _ => unreachable!("no throwing stub for {function}/{environment}"),
        };
        self.parse_expression(source)
    }

    /// Rewrites `createServerOnlyFn(fn)` / `createClientOnlyFn(fn)`. Returns
    /// `true` if `expression` was a matching call (and was replaced).
    fn rewrite_env_only(&mut self, expression: &mut Expression<'a>) -> bool {
        let Expression::CallExpression(call) = expression else {
            return false;
        };
        let Expression::Identifier(callee) = &call.callee else {
            return false;
        };
        let kind = match self.env_fn(callee) {
            Some(kind @ (EnvFn::ServerOnly | EnvFn::ClientOnly)) => kind,
            _ => return false,
        };
        let keep = matches!(
            (kind, self.target),
            (EnvFn::ServerOnly, Target::Server) | (EnvFn::ClientOnly, Target::Client)
        );
        if keep {
            // Replace the whole call with its inner function argument.
            let Some(inner) = call
                .arguments
                .first_mut()
                .and_then(|argument| argument.as_expression_mut())
            else {
                return false;
            };
            *expression = inner.take_in(&self.allocator);
        } else {
            let (function, environment) = match kind {
                EnvFn::ServerOnly => ("createServerOnlyFn", "server"),
                EnvFn::ClientOnly => ("createClientOnlyFn", "client"),
                EnvFn::Isomorphic | EnvFn::Middleware => unreachable!(),
            };
            *expression = self.throwing_stub(function, environment);
        }
        true
    }

    /// Validates that `expression` is a complete
    /// `createIsomorphicFn()[.client(_)][.server(_)]` chain (read-only).
    fn is_isomorphic_chain(&self, expression: &Expression<'a>) -> bool {
        let Expression::CallExpression(call) = expression else {
            return false;
        };
        match &call.callee {
            Expression::Identifier(callee) => {
                self.env_fn(callee) == Some(EnvFn::Isomorphic) && call.arguments.is_empty()
            }
            Expression::StaticMemberExpression(member) => {
                matches!(member.property.name.as_str(), "client" | "server")
                    && self.is_isomorphic_chain(&member.object)
            }
            _ => false,
        }
    }

    /// Extracts the `.client` / `.server` implementation arguments from a
    /// validated isomorphic chain, consuming the chain.
    fn extract_isomorphic(
        &self,
        expression: &mut Expression<'a>,
        client: &mut Option<Expression<'a>>,
        server: &mut Option<Expression<'a>>,
    ) {
        let Expression::CallExpression(call) = expression else {
            return;
        };
        // Take the method argument before borrowing `callee`, so the two
        // disjoint field borrows never overlap.
        let argument = call
            .arguments
            .first_mut()
            .and_then(|argument| argument.as_expression_mut())
            .map(|argument| argument.take_in(&self.allocator));
        let Expression::StaticMemberExpression(member) = &mut call.callee else {
            return;
        };
        self.extract_isomorphic(&mut member.object, client, server);
        match member.property.name.as_str() {
            "client" => *client = argument,
            "server" => *server = argument,
            _ => {}
        }
    }

    /// Rewrites a full isomorphic chain to the target's implementation. Returns
    /// `true` if `expression` was such a chain.
    fn rewrite_isomorphic(&mut self, expression: &mut Expression<'a>) -> bool {
        // Only the outermost chain node (its callee is a `.client`/`.server`
        // member) is a rewrite point; the bare `createIsomorphicFn()` base is
        // left for its enclosing member call to consume.
        let is_chain_tail = matches!(
            expression,
            Expression::CallExpression(call)
                if matches!(&call.callee, Expression::StaticMemberExpression(_))
        );
        if !is_chain_tail || !self.is_isomorphic_chain(expression) {
            return false;
        }
        let mut client = None;
        let mut server = None;
        self.extract_isomorphic(expression, &mut client, &mut server);
        let chosen = match self.target {
            Target::Client => client,
            // ReactServer is server-like for the isomorphic chain. Unreachable in
            // practice (`apply_env_transform` returns early for non-`Client`), but
            // required for the match to be exhaustive.
            Target::Server | Target::ReactServer => server,
        };
        *expression = chosen.unwrap_or_else(|| self.parse_expression("(() => {})"));
        true
    }

    /// Whether `expression` is a `createMiddleware()[.method(_)]*` chain.
    fn is_middleware_chain(&self, expression: &Expression<'a>) -> bool {
        let Expression::CallExpression(call) = expression else {
            return false;
        };
        match &call.callee {
            Expression::Identifier(callee) => {
                self.env_fn(callee) == Some(EnvFn::Middleware) && call.arguments.is_empty()
            }
            Expression::StaticMemberExpression(member) => {
                self.is_middleware_chain(&member.object)
            }
            _ => false,
        }
    }

    /// Strips the environment-specific method calls from a validated
    /// `createMiddleware` chain, mirroring `handleCreateMiddleware`: on the
    /// client the `.server(...)`, `.validator(...)` and `.inputValidator(...)`
    /// calls are removed (severing their references to server-only code), while
    /// `.middleware(...)` and `.client(...)` are kept. Operates bottom-up so a
    /// stripped level is spliced out cleanly.
    fn strip_middleware(&mut self, expression: &mut Expression<'a>) {
        let Expression::CallExpression(call) = expression else {
            return;
        };
        let Expression::StaticMemberExpression(member) = &mut call.callee else {
            return;
        };
        self.strip_middleware(&mut member.object);
        let strip = matches!(
            member.property.name.as_str(),
            "server" | "validator" | "inputValidator"
        );
        if strip {
            let object = member.object.take_in(&self.allocator);
            *expression = object;
            self.changed = true;
        }
    }
}

impl<'a> VisitMut<'a> for EnvTransform<'a, '_> {
    fn visit_expression(&mut self, expression: &mut Expression<'a>) {
        if self.rewrite_env_only(expression) || self.rewrite_isomorphic(expression) {
            self.changed = true;
            // Descend into the replacement so a nested directive helper (e.g. an
            // isomorphic impl that itself calls a server-only fn) is handled too.
            self.visit_expression(expression);
            return;
        }
        if self.is_middleware_chain(expression) {
            // Strip the server-only method calls, then descend into what remains
            // (kept `.client`/`.middleware` arguments may contain their own
            // directive helpers). Re-visiting the whole node instead would loop,
            // since a stripped chain is still a chain.
            self.strip_middleware(expression);
            walk_mut::walk_expression(self, expression);
            return;
        }
        walk_mut::walk_expression(self, expression);
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

    /// The rewritten-source rule at the transform boundary: a `"use client"`
    /// module built for the react-server graph is parsed from text diffpack
    /// GENERATED (the client-reference re-exports), so its map says so and
    /// carries that text — never a silent claim on the component file.
    #[test]
    fn a_directive_rewritten_module_labels_its_map_as_generated_and_carries_that_text() {
        let source = "\"use client\";\nimport { useState } from 'react';\nexport function Counter(){ const [n, s] = useState(0); return n; }\n";
        let path = Path::new("/app/src/Counter.tsx");
        let rewritten = transform_module_in_language(
            path,
            source,
            Target::ReactServer,
            false,
            JsxExtensions::default(),
            &ProjectConfig::default(),
            SourceLanguage::FromPath,
            true,
        );
        assert!(rewritten.diagnostics.is_empty(), "{:?}", rewritten.diagnostics);
        let map = rewritten.map.expect("a map was requested");
        assert_eq!(
            map.origin(),
            MapOrigin::Generated("rsc-directive"),
            "the RSC directive rewrite must be named as the source the positions index"
        );
        let module_source: std::sync::Arc<str> = std::sync::Arc::from(source);
        let text = map.source_text(&module_source);
        assert!(
            text.contains("createClientModuleProxy") && !text.contains("useState"),
            "the map must carry the REWRITTEN text its positions were measured \
             against, got: {text}"
        );

        // The same module built for the client graph is parsed from the file
        // itself, so its map claims the file — correctly.
        let untouched = transform_module_in_language(
            path,
            source,
            Target::Client,
            false,
            JsxExtensions::default(),
            &ProjectConfig::default(),
            SourceLanguage::FromPath,
            true,
        );
        assert_eq!(
            untouched.map.expect("a map was requested").origin(),
            MapOrigin::File,
        );
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
        assert!(result.map.is_none(), "no map was requested, so none is invented");
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
    fn react_server_target_replaces_use_client_module_with_client_references() {
        // The GATE for RSC Slice R1: a `"use client"` module built for the
        // react-server target must come out as react-server-dom client references
        // (createClientModuleProxy re-exports), with NONE of its component code
        // reaching the react-server graph — and this must be driven by the Target,
        // through the full transform pipeline, not just the rsc.rs helper.
        let source = "\"use client\";\nimport { useState } from 'react';\nexport function Counter(){ const [n, s] = useState(0); return n; }\nexport default Counter;";
        let path = Path::new("/app/src/Counter.tsx");

        let react_server = transform_module(path, source, Target::ReactServer);
        assert!(
            react_server.diagnostics.is_empty(),
            "{:?}",
            react_server.diagnostics
        );
        // The client reference surface is present...
        assert!(
            react_server.code.contains("createClientModuleProxy"),
            "react-server build must emit client references: {}",
            react_server.code
        );
        assert!(
            react_server.code.contains("react-server-dom-webpack/server"),
            "must import the proxy from react-server-dom: {}",
            react_server.code
        );
        // ...and the component body is NOT (no `useState` reaches the server graph).
        assert!(
            !react_server.code.contains("useState"),
            "react-server build must not ship client component code: {}",
            react_server.code
        );
        // The react-server-dom module import is collected as a real dependency by
        // the normal parse of the rewritten source.
        assert!(
            react_server
                .dependencies
                .iter()
                .any(|dep| dep == "react-server-dom-webpack/server"),
            "the client-reference proxy import must be a real dep: {:?}",
            react_server.dependencies
        );

        // Contrast: the SAME module built for a non-RSC target keeps the real
        // component code (proving the transform is gated on Target::ReactServer).
        let client = transform_module(path, source, Target::Client);
        assert!(
            client.code.contains("useState") && !client.code.contains("createClientModuleProxy"),
            "non-react-server target keeps the component code: {}",
            client.code
        );
    }

    #[test]
    fn react_server_target_registers_use_server_module_as_server_references() {
        // In the react-server graph a `"use server"` module keeps its real body and
        // registers each export via `registerServerReference` (RSC Slice R2/C).
        let result = transform_module(
            Path::new("/app/src/actions.ts"),
            "\"use server\";\nexport async function increment(n){ return n + 1 }",
            Target::ReactServer,
        );
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert!(
            result.code.contains("registerServerReference"),
            "react-server 'use server' registers server references: {}",
            result.code
        );
        assert!(result.code.contains("return n + 1"), "real body kept: {}", result.code);
        // The react-server-dom-webpack/server import is collected as a real dep.
        assert!(
            result
                .dependencies
                .iter()
                .any(|d| d == "react-server-dom-webpack/server"),
            "the server writer import must be a collected dependency: {:?}",
            result.dependencies
        );
    }

    /// The SSR graph is a CLIENT layer that happens to run on the server, so a
    /// `"use server"` module there is a reference stub, never the real handler.
    ///
    /// This is not a preference. `registerServerReference` lives in the flight WRITER
    /// (`react-server-dom-webpack/server`), whose `exports` map names a real file only
    /// under the `react-server` condition; under every other condition it resolves to a
    /// module whose whole body is `throw new Error("The React Server Writer cannot be
    /// used outside a react-server environment")`. Emitting the react-server transform
    /// into the SSR graph put that throw on the load path of any client component that
    /// imports an action, killing the render before a single byte of HTML.
    #[test]
    fn the_ssr_server_target_gets_reference_stubs_not_the_flight_writer() {
        let source = "\"use server\";\nexport async function increment(n){ return n + 1 }";
        let result = transform_module(Path::new("/app/src/actions.ts"), source, Target::Server);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert!(
            !result.code.contains("react-server-dom-webpack/server"),
            "the SSR graph must never import the flight writer: {}",
            result.code
        );
        assert!(
            !result
                .dependencies
                .iter()
                .any(|d| d == "react-server-dom-webpack/server"),
            "nor collect it as a dependency: {:?}",
            result.dependencies
        );
        assert!(
            result.code.contains("createServerReference"),
            "it gets the same reference stub the browser gets: {}",
            result.code
        );
        assert!(
            !result.code.contains("return n + 1"),
            "the real handler body stays out of the SSR graph: {}",
            result.code
        );
        // The id the stub carries is byte-identical to the one the react-server graph
        // registers, so a form action posted from an SSR-rendered page still lands on
        // the real handler.
        let react_server = transform_module(
            Path::new("/app/src/actions.ts"),
            source,
            Target::ReactServer,
        );
        let id = crate::rsc::action_reference_id(Path::new("/app/src/actions.ts"), "increment");
        assert!(result.code.contains(&id), "stub id {id}: {}", result.code);
        assert!(
            react_server.code.contains("registerServerReference"),
            "only the react-server graph registers: {}",
            react_server.code
        );
    }

    #[test]
    fn client_target_replaces_use_server_module_with_rpc_stubs() {
        // The client build drops the server body and emits createServerReference stubs.
        let result = transform_module(
            Path::new("/app/src/actions.ts"),
            "\"use server\";\nimport { readFile } from 'node:fs/promises';\nexport async function increment(n){ await readFile('x'); return n + 1 }",
            Target::Client,
        );
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert!(!result.code.contains("node:fs"), "no server import ships: {}", result.code);
        assert!(!result.code.contains("return n + 1"), "no server body ships: {}", result.code);
        assert!(
            result.code.contains("createServerReference"),
            "client emits a server-reference stub: {}",
            result.code
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
        assert!(transformed.diagnostics.is_empty(), "{:?}", transformed.diagnostics);
        let code = &transformed.code;
        // Same-line space between two elements survives as a child of its own.
        assert!(
            code.contains("\"b\", { children: \"x\" }),\n\t\" \",")
                || code.contains("\" \","),
            "the space between <b> and <i> must be preserved: {code}"
        );
        // Indentation-only children (they contain a newline) are dropped.
        assert!(!code.contains("\"\\n"), "indentation must not leak into children: {code}");
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
        assert!(transformed.code.contains("require.esm(\"react/jsx-runtime\")"));
    }

    /// A `.vue` file's extension says nothing about the language of what its
    /// compiler emitted: `@vue/compiler-sfc` leaves a `<script lang="ts">`
    /// component's annotations in place for the bundler to strip (which is why
    /// plugin-vue hands its own output to Vite with `lang: "ts"`). Parsing that
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
        assert!(typescript.diagnostics.is_empty(), "{:?}", typescript.diagnostics);
        assert!(!typescript.code.contains("_ctx: any"), "{}", typescript.code);
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
        assert!(transformed.diagnostics.is_empty(), "{:?}", transformed.diagnostics);
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
        assert!(transformed.diagnostics.is_empty(), "{:?}", transformed.diagnostics);
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
        assert!(with_pragma.diagnostics.is_empty(), "{:?}", with_pragma.diagnostics);
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

    /// The Next default page: JSX in a `.js` file. It must compile, and — the part
    /// that made the defect so damaging — it must yield its DEPENDENCIES, because a
    /// fatal parse returns a dummy program and the importer's whole subtree
    /// (`components/Gallery.js`, ...) silently vanishes from the graph.
    #[test]
    fn next_compiles_jsx_in_a_js_module_and_still_sees_its_imports() {
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
            JsxExtensions::NextJs,
            &JsxConfig::default(),
        );

        assert!(transformed.diagnostics.is_empty(), "{:?}", transformed.diagnostics);
        assert!(
            transformed.code.contains("react/jsx-runtime"),
            "JSX must be lowered through the automatic runtime: {}",
            transformed.code
        );
        assert_eq!(transformed.dependencies, ["../components/Gallery", "react/jsx-runtime"]);
    }

    /// The counterpart: under Vite/esbuild the same file is a hard error, and the
    /// message has to say why and what to do — not oxc's bare "Unexpected JSX
    /// expression". This is what stops the Next fix from becoming "JSX everywhere".
    #[test]
    fn a_vite_js_module_with_jsx_is_a_named_actionable_error() {
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
        assert!(message.contains("JSX is not enabled for `.js` files"), "{message}");
        assert!(message.contains("main.jsx"), "{message}");
        assert!(
            !message.contains("Unexpected JSX expression"),
            "the raw oxc message must be replaced: {message}"
        );
    }

    /// The rule is scoped, not a blanket "JSX everywhere": in TypeScript `<T>x` is a
    /// type assertion, so a `.ts` module stays JSX-free even under Next.
    #[test]
    fn a_next_ts_module_still_reads_angle_brackets_as_a_type_assertion() {
        let transformed = transform_module_with_options(
            Path::new("lib/cast.ts"),
            "const value: unknown = 1; export const text = <string>value;",
            Target::Server,
            false,
            JsxExtensions::NextJs,
            &JsxConfig::default(),
        );

        assert!(transformed.diagnostics.is_empty(), "{:?}", transformed.diagnostics);
        assert!(
            !transformed.code.contains("jsx-runtime"),
            "a `.ts` type assertion must not become a JSX element: {}",
            transformed.code
        );
    }

    /// The auxiliary directive probe runs on a SEPARATE parse. When it did not know
    /// `.js` could hold JSX it answered "no directive" for a JSX-bearing
    /// `"use client"` module — a silent wrong answer once the main parse succeeds.
    #[test]
    fn a_use_client_directive_survives_jsx_in_a_js_module() {
        let source = r#""use client";
            export default function Counter() { return <button>+</button>; }
        "#;
        assert_eq!(
            crate::rsc::detect_directive(Path::new("/app/components/Counter.js"), source),
            Some(crate::rsc::RscDirective::Client)
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

    fn demand_names<'a>(
        transformed: &'a TransformResult,
        specifier: &str,
    ) -> Option<&'a DependencyDemand> {
        transformed
            .dependency_demands
            .iter()
            .find(|demand| demand.specifier == specifier)
    }

    #[test]
    fn client_build_neutralizes_server_only_fn_and_drops_its_server_import() {
        // Mirrors `@tanstack/start-client-core`'s getStartContextServerOnly.js:
        // a server-only wrapper around a value imported from a server-only
        // package. On the client the wrapper throws and the reference to
        // `getStartContext` is severed, so the server import is no longer
        // demanded and is pruned by the side-effect-free tree-shaking.
        let source = r#"
            import { createServerOnlyFn } from "@tanstack/start-fn-stubs";
            import { getStartContext } from "@tanstack/start-storage-context";
            export const getStartContextServerOnly = createServerOnlyFn(getStartContext);
        "#;

        let client = transform_module(Path::new("mod.js"), source, Target::Client);
        assert!(client.diagnostics.is_empty(), "{:?}", client.diagnostics);
        assert!(
            client.code.contains("can only be called on the server"),
            "client build must emit the throwing stub: {}",
            client.code
        );
        // The server-only value's package is no longer demanded on the client.
        let storage = demand_names(&client, "@tanstack/start-storage-context").unwrap();
        assert!(
            !storage.all && storage.names.is_empty(),
            "client build must not demand the server storage package: {storage:?}"
        );

        // The server build keeps the neutral stub call and its import demand.
        let server = transform_module(Path::new("mod.js"), source, Target::Server);
        let storage = demand_names(&server, "@tanstack/start-storage-context").unwrap();
        assert_eq!(
            storage.names,
            ["getStartContext"],
            "server build must keep demanding getStartContext"
        );
    }

    #[test]
    fn client_build_collapses_isomorphic_fn_to_client_impl() {
        // Mirrors getRouterInstance.js: an isomorphic fn whose server branch is
        // the only user of a server import. On the client it collapses to the
        // client impl, dropping the server import entirely.
        let source = r#"
            import { createIsomorphicFn } from "@tanstack/start-fn-stubs";
            import { getStartContext } from "@tanstack/start-storage-context";
            export const getRouterInstance = createIsomorphicFn()
                .client(() => window.__TSR_ROUTER__)
                .server(() => getStartContext().getRouter());
        "#;

        let client = transform_module(Path::new("mod.js"), source, Target::Client);
        assert!(client.diagnostics.is_empty(), "{:?}", client.diagnostics);
        assert!(
            client.code.contains("__TSR_ROUTER__"),
            "client impl must survive: {}",
            client.code
        );
        assert!(
            !client.code.contains("getStartContext"),
            "the server impl's reference to getStartContext must be gone on the client: {}",
            client.code
        );
        assert!(
            demand_names(&client, "@tanstack/start-storage-context").is_none_or(
                |demand| !demand.all && demand.names.is_empty()
            ),
            "client build must not pull the server storage package"
        );

        let server = transform_module(Path::new("mod.js"), source, Target::Server);
        let storage = demand_names(&server, "@tanstack/start-storage-context").unwrap();
        assert_eq!(storage.names, ["getStartContext"]);
    }

    #[test]
    fn env_transform_ignores_same_named_local_binding() {
        // A user's own `createServerOnlyFn` (not a @tanstack import) must never
        // be rewritten.
        let source = r#"
            const createServerOnlyFn = (fn) => fn;
            export const value = createServerOnlyFn(() => 1);
        "#;
        let client = transform_module(Path::new("mod.js"), source, Target::Client);
        assert!(
            !client.code.contains("can only be called on the server"),
            "a local same-named binding must not be treated as the directive helper: {}",
            client.code
        );
    }
}

