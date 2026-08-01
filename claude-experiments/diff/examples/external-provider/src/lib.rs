//! Executable example of embedding Diffpack without depending on its CLI crate.

use std::path::PathBuf;

use diffpack_core::{
    EmittedAsset, LoadRequest, LoadedSource, ModuleProvider, ProviderDiagnostic, ResolveRequest,
    ResolveResult, SourceLanguage, TransformOutput, TransformRequest,
};

pub struct ExampleProvider {
    pub watch_file: PathBuf,
}

impl ModuleProvider for ExampleProvider {
    fn name(&self) -> &str {
        "example:provider"
    }

    fn resolve(&self, request: ResolveRequest<'_>) -> Result<ResolveResult, ProviderDiagnostic> {
        Ok(match request.specifier {
            "custom:answer" => ResolveResult::Resolved("virtual:answer".into()),
            "custom:fatal" => ResolveResult::Resolved("virtual:fatal".into()),
            "external:host-api" => ResolveResult::External("host-api".into()),
            _ => ResolveResult::NoMatch,
        })
    }

    fn load(&self, request: LoadRequest<'_>) -> Result<Option<LoadedSource>, ProviderDiagnostic> {
        if request.id == "virtual:fatal" {
            return Err(ProviderDiagnostic {
                message: "intentional fatal diagnostic".into(),
                provider: Some(self.name().into()),
            });
        }
        Ok((request.id == "virtual:answer").then(|| LoadedSource {
            code: format!(
                "const answer: number = 42; export const config = {:?}; export default answer;",
                std::fs::read_to_string(&self.watch_file).unwrap_or_default()
            )
            .into_bytes(),
            language: SourceLanguage::TypeScript,
            source_map: None,
            watch_files: vec![self.watch_file.clone()],
            diagnostics: Vec::new(),
        }))
    }

    fn transform(
        &self,
        request: TransformRequest<'_>,
    ) -> Result<Option<TransformOutput>, ProviderDiagnostic> {
        if !request.id.ends_with("entry.js") {
            return Ok(None);
        }
        let code = String::from_utf8_lossy(request.code)
            .replace("__BUILD_LABEL__", "\"external-provider\"");
        Ok(Some(TransformOutput {
            code: code.into_bytes(),
            language: request.language,
            source_map: None,
            watch_files: vec![self.watch_file.clone()],
            emitted_assets: vec![EmittedAsset {
                name: Some("provider-note.txt".into()),
                source: b"emitted by example provider".to_vec(),
            }],
            diagnostics: vec![diffpack_core::ProviderMessage {
                message: "example provider transformed the entry".into(),
                fatal: false,
            }],
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use diffpack_core::{BuildMode, Environment, Platform};
    use diffpack_default_loader::BuildEngine;

    #[test]
    fn external_provider_runs_end_to_end_and_rebuilds_its_watch_file() {
        let directory = tempfile::tempdir().unwrap();
        let watch = directory.path().join("provider.config");
        std::fs::write(&watch, "one").unwrap();
        std::fs::write(
            directory.path().join("entry.js"),
            "import answer from 'custom:answer'; import 'external:host-api'; export const label = __BUILD_LABEL__; export default answer;",
        )
        .unwrap();

        let watch = watch.canonicalize().unwrap();
        let engine = BuildEngine::builder(directory.path())
            .environment(Environment {
                name: "example".into(),
                platform: Platform::Node,
                mode: BuildMode::Development,
            })
            .provider(ExampleProvider {
                watch_file: watch.clone(),
            })
            .build()
            .unwrap();
        let (mut bundler, initial) = engine.discover("entry.js").unwrap();
        assert!(initial.transformed_modules > 0);
        assert!(initial.diagnostics.iter().any(|diagnostic| {
            !diagnostic.is_fatal()
                && diagnostic
                    .message
                    .contains("example provider transformed the entry")
        }));
        std::fs::write(&watch, "two").unwrap();
        let rebuilt = bundler.rebuild_path(&watch).unwrap();
        assert!(!rebuilt.delta.changed.is_empty() || rebuilt.transformed_modules > 0);
    }

    #[test]
    fn fatal_provider_diagnostic_stops_discovery_with_provider_ownership() {
        let directory = tempfile::tempdir().unwrap();
        let watch = directory.path().join("provider.config");
        std::fs::write(&watch, "one").unwrap();
        std::fs::write(directory.path().join("entry.js"), "import 'custom:fatal';").unwrap();
        let engine = BuildEngine::builder(directory.path())
            .provider(ExampleProvider {
                watch_file: watch.canonicalize().unwrap(),
            })
            .build()
            .unwrap();
        let error = match engine.discover("entry.js") {
            Ok(_) => panic!("fatal provider unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(error.contains("example:provider"), "{error}");
        assert!(error.contains("intentional fatal diagnostic"), "{error}");
    }
}
