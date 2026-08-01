//! Linker-facing JavaScript transform records and compilation configuration.

use std::path::Path;

use oxc_span::SourceType;
use oxc_transformer::{JsxRuntime as OxcJsxRuntime, TransformOptions};

use crate::parser::JsxExtensions;
use crate::source_map::{LineTrack, ModuleSourceMap};

/// The environment a module is being compiled for.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum Target {
    Client,
    #[default]
    Server,
    IsolatedServer,
}

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
/// against the compiler's default JSX runtime. The layers that fill it in, weakest
/// first, are project settings, then host settings, then a per-file `@jsxImportSource` /
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
    /// field, not whole-record, so partial host settings only replace their
    /// configured counterparts.
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
    pub fn apply(&self, options: &mut TransformOptions) {
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
    pub fn apply(&self, options: &mut TransformOptions) {
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

impl FoldExpression {
    /// Renders the folded expression as JavaScript without changing its value.
    pub fn to_javascript(&self) -> String {
        match self {
            Self::Number(bits) => format_javascript_number(f64::from_bits(*bits))
                .unwrap_or_else(|| "<non-finite>".into()),
            Self::Reference(name) => name.clone(),
            Self::Add(left, right) => {
                format!("({} + {})", left.to_javascript(), right.to_javascript())
            }
        }
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
    pub fn source_type(self, path: &Path, jsx: JsxExtensions) -> SourceType {
        match self {
            SourceLanguage::FromPath => crate::parser::source_type_for(path, jsx),
            SourceLanguage::JavaScript => SourceType::default().with_module(true),
            SourceLanguage::TypeScript => SourceType::default()
                .with_typescript(true)
                .with_module(true),
        }
    }
}
