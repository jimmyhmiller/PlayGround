//! Framework-independent contracts for Diffpack's module graph.
//!
//! This crate is intentionally free of host filesystem, package-loader,
//! framework, and stylesheet policy. Integrations contribute contents through [`ModuleProvider`];
//! the graph remains authoritative for identity, dependencies, invalidation, and
//! chunk ownership.

use std::path::{Path, PathBuf};

pub mod async_graph;
pub mod build_profile;
pub mod bundle;
mod cancel;
pub mod compiler;
pub mod dead_branch;
pub mod diagnostic;
mod emission;
pub mod frontend_profile;
pub mod graph;
pub mod js_reachability;
pub mod linker;
pub mod memory;
pub mod minify;
pub mod module_graph;
pub mod parser;
pub mod resource_id;
pub mod runtime;
pub mod source_map;
pub mod text_edit;
pub mod transform;
pub mod tree_shake;
mod visualization;

pub use cancel::CancelToken;
pub use diagnostic::{Diagnostic, DiagnosticKind, partition_diagnostics};
pub use emission::{EmitOptions, ModuleFormat};
pub use graph::{BuildUpdate, DirectReachability, DirectReachabilityUpdate, GraphDelta};
pub use resource_id::ResourceId;
pub use visualization::{VisualizationEdge, VisualizationGraph, VisualizationNode};

/// Stable identity of a module in the graph, including any meaningful query.
pub type ModuleId = String;

/// The environment for which a module is being built.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Environment {
    pub name: String,
    pub platform: Platform,
    pub mode: BuildMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Platform {
    Browser,
    Node,
    Neutral,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuildMode {
    Development,
    Production,
}

/// Information shared by resolve, load, and transform hooks.
#[derive(Debug, Clone)]
pub struct HookContext<'a> {
    pub environment: &'a Environment,
    pub project_root: &'a Path,
}

#[derive(Debug, Clone)]
pub struct ResolveRequest<'a> {
    pub specifier: &'a str,
    pub importer: Option<&'a str>,
    pub context: HookContext<'a>,
}

/// A provider either declines a request, resolves it, or marks it external.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveResult {
    NoMatch,
    Resolved(ModuleId),
    External(String),
}

#[derive(Debug, Clone)]
pub struct LoadRequest<'a> {
    pub id: &'a str,
    pub context: HookContext<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceLanguage {
    JavaScript,
    TypeScript,
    Json,
    Text,
    Binary,
}

/// Source supplied to the core before parsing and dependency extraction.
#[derive(Debug, Clone)]
pub struct LoadedSource {
    pub code: Vec<u8>,
    pub language: SourceLanguage,
    pub source_map: Option<Vec<u8>>,
    pub watch_files: Vec<PathBuf>,
    pub diagnostics: Vec<ProviderMessage>,
}

#[derive(Debug, Clone)]
pub struct TransformRequest<'a> {
    pub id: &'a str,
    pub code: &'a [u8],
    pub language: SourceLanguage,
    pub context: HookContext<'a>,
}

/// One ordered transform's output. `None` means the transform did not apply.
#[derive(Debug, Clone)]
pub struct TransformOutput {
    pub code: Vec<u8>,
    pub language: SourceLanguage,
    pub source_map: Option<Vec<u8>>,
    pub watch_files: Vec<PathBuf>,
    pub emitted_assets: Vec<EmittedAsset>,
    pub diagnostics: Vec<ProviderMessage>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderMessage {
    pub message: String,
    pub fatal: bool,
}

#[derive(Debug, Clone)]
pub struct EmittedAsset {
    pub name: Option<String>,
    pub source: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderDiagnostic {
    pub message: String,
    pub provider: Option<String>,
}

impl ProviderDiagnostic {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            provider: None,
        }
    }
}

/// Native extension seam for module resolution and compilation.
///
/// Providers are ordered. Resolution and loading stop at the first non-empty
/// answer; every applicable transform runs in order. Lifecycle/output hooks are
/// deliberately deferred until the chunk graph becomes a public core type.
pub trait ModuleProvider: Send + Sync {
    fn name(&self) -> &str;

    fn resolve(&self, _request: ResolveRequest<'_>) -> Result<ResolveResult, ProviderDiagnostic> {
        Ok(ResolveResult::NoMatch)
    }

    fn load(&self, _request: LoadRequest<'_>) -> Result<Option<LoadedSource>, ProviderDiagnostic> {
        Ok(None)
    }

    fn transform(
        &self,
        _request: TransformRequest<'_>,
    ) -> Result<Option<TransformOutput>, ProviderDiagnostic> {
        Ok(None)
    }
}

/// Immutable ordered provider collection suitable for sharing across frontend
/// workers. It centralizes ordering semantics before the existing loader is
/// migrated onto the contract.
#[derive(Clone, Default)]
pub struct ProviderPipeline {
    providers: std::sync::Arc<[Box<dyn ModuleProvider>]>,
}

impl ProviderPipeline {
    pub fn new(providers: Vec<Box<dyn ModuleProvider>>) -> Self {
        Self {
            providers: providers.into(),
        }
    }

    pub fn providers(&self) -> impl Iterator<Item = &dyn ModuleProvider> {
        self.providers.iter().map(Box::as_ref)
    }

    pub fn resolve(
        &self,
        request: ResolveRequest<'_>,
    ) -> Result<ResolveResult, ProviderDiagnostic> {
        for provider in self.providers.iter() {
            let result = provider.resolve(request.clone())?;
            if result != ResolveResult::NoMatch {
                return Ok(result);
            }
        }
        Ok(ResolveResult::NoMatch)
    }

    pub fn load(
        &self,
        request: LoadRequest<'_>,
    ) -> Result<Option<LoadedSource>, ProviderDiagnostic> {
        for provider in self.providers.iter() {
            if let Some(source) = provider.load(request.clone())? {
                return Ok(Some(source));
            }
        }
        Ok(None)
    }

    pub fn transform(
        &self,
        id: &str,
        mut source: LoadedSource,
        context: HookContext<'_>,
    ) -> Result<(LoadedSource, Vec<EmittedAsset>), ProviderDiagnostic> {
        let mut assets = Vec::new();
        for provider in self.providers.iter() {
            let request = TransformRequest {
                id,
                code: &source.code,
                language: source.language,
                context: context.clone(),
            };
            if let Some(output) = provider.transform(request)? {
                source.code = output.code;
                source.language = output.language;
                source.source_map = output.source_map;
                source.watch_files.extend(output.watch_files);
                assets.extend(output.emitted_assets);
                source.diagnostics.extend(output.diagnostics);
            }
        }
        source.watch_files.sort();
        source.watch_files.dedup();
        Ok((source, assets))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Resolver(&'static str, ResolveResult);

    impl ModuleProvider for Resolver {
        fn name(&self) -> &str {
            self.0
        }

        fn resolve(
            &self,
            _request: ResolveRequest<'_>,
        ) -> Result<ResolveResult, ProviderDiagnostic> {
            Ok(self.1.clone())
        }
    }

    #[test]
    fn resolution_stops_at_the_first_provider_that_matches() {
        let pipeline = ProviderPipeline::new(vec![
            Box::new(Resolver("pre", ResolveResult::NoMatch)),
            Box::new(Resolver(
                "normal",
                ResolveResult::Resolved("virtual:x".into()),
            )),
            Box::new(Resolver("post", ResolveResult::Resolved("wrong".into()))),
        ]);
        let environment = Environment {
            name: "client".into(),
            platform: Platform::Browser,
            mode: BuildMode::Production,
        };
        let root = Path::new("/project");
        let result = pipeline
            .resolve(ResolveRequest {
                specifier: "x",
                importer: None,
                context: HookContext {
                    environment: &environment,
                    project_root: root,
                },
            })
            .unwrap();
        assert_eq!(result, ResolveResult::Resolved("virtual:x".into()));
    }
}
