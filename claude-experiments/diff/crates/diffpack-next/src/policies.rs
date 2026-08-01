//! Next-owned runtime and special-module policies.

use std::path::Path;

#[derive(Debug, Default, Clone, Copy)]
pub struct NextRuntimePolicy;

impl diffpack_core::runtime::RuntimeIntegrationPolicy for NextRuntimePolicy {
    fn configure(
        &self,
        request: diffpack_core::runtime::RuntimePolicyRequest<'_>,
    ) -> diffpack_core::runtime::RuntimePolicyOutput {
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
        diffpack_core::runtime::RuntimePolicyOutput {
            compatibility_prelude,
            ..Default::default()
        }
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
