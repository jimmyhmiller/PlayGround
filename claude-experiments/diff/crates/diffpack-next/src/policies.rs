//! Next-owned runtime and special-module policies.

use std::path::Path;

#[derive(Debug, Default, Clone, Copy)]
pub struct NextRuntimePolicy;

impl diffpack_core::runtime::RuntimeIntegrationPolicy for NextRuntimePolicy {
    fn configure(
        &self,
        request: diffpack_core::runtime::RuntimePolicyRequest<'_>,
    ) -> Result<diffpack_core::runtime::RuntimePolicyOutput, String> {
        let compatibility_prelude = (request.is_main
            && request.format == diffpack_core::ModuleFormat::BrowserEsm)
            .then(|| {
                crate::rsc::webpack_runtime_seam_for_modules(
                    request
                        .modules
                        .iter()
                        .map(|module| (module.id, module.source)),
                    request.entry_id,
                    request.base,
                    request.chunk_files,
                )
            })
            .flatten();
        Ok(diffpack_core::runtime::RuntimePolicyOutput {
            compatibility_prelude: compatibility_prelude.map(|value| {
                diffpack_core::runtime::RuntimeContribution::new(
                    "framework-compatibility",
                    "diffpack-next",
                    value,
                )
            }),
            ..Default::default()
        })
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NextSpecialModulePolicy;

impl diffpack_default_loader::module_policy::SpecialModulePolicy for NextSpecialModulePolicy {
    fn asset_module(
        &self,
        path: &Path,
        bytes: &[u8],
        public_name: &str,
        base: &str,
        responsive_variants: bool,
        compile: &mut diffpack_default_loader::module_policy::SyntheticCompiler<'_>,
    ) -> Result<Option<diffpack_default_loader::module::SpecialModule>, String> {
        crate::static_image::module(
            path,
            bytes,
            public_name,
            base,
            responsive_variants,
            |source| compile(Path::new("diffpack-image-import.js"), source),
        )
    }
}

#[cfg(test)]
mod tests {
    use diffpack_core::runtime::{
        RuntimeIntegrationPolicy, RuntimePolicyChain, RuntimePolicyModule, RuntimePolicyRequest,
    };

    use super::NextRuntimePolicy;

    fn snapshot(format: diffpack_core::ModuleFormat) -> Vec<String> {
        let modules = [RuntimePolicyModule {
            id: "client.js",
            source: "'use client'; export default function Client(){}",
        }];
        RuntimePolicyChain::new(vec![
            std::sync::Arc::new(NextRuntimePolicy),
            std::sync::Arc::new(diffpack_web::policies::WebRuntimePolicy),
        ])
        .configure(RuntimePolicyRequest {
            format,
            is_main: true,
            hmr: false,
            entry_id: "entry",
            entry_runtime_id: 0,
            any_async: false,
            base: "/",
            chunk_files: &[],
            modules: &modules,
            browser_process_shim: true,
        })
        .unwrap()
        .describe()
    }

    #[test]
    fn next_client_and_server_runtime_profile_snapshots() {
        assert_eq!(
            snapshot(diffpack_core::ModuleFormat::BrowserEsm),
            [
                "browser-process-compatibility@diffpack-web",
                "framework-compatibility@diffpack-next",
                "browser-require-native@diffpack-web",
            ]
        );
        assert_eq!(
            snapshot(diffpack_core::ModuleFormat::Esm),
            ["browser-require-native@diffpack-web"]
        );
    }
}
