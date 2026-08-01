//! TanStack-owned runtime, special-module, and output policies.

#[derive(Debug, Default, Clone, Copy)]
pub struct TanStackRuntimePolicy;

impl diffpack_core::runtime::RuntimeIntegrationPolicy for TanStackRuntimePolicy {
    fn configure(
        &self,
        request: diffpack_core::runtime::RuntimePolicyRequest<'_>,
    ) -> diffpack_core::runtime::RuntimePolicyOutput {
        let mut output = diffpack_core::runtime::RuntimePolicyOutput::default();
        if request.is_main
            && request.format == diffpack_core::ModuleFormat::BrowserEsm
            && request.browser_process_shim
        {
            output
                .entry_preludes
                .push(crate::runtime::BROWSER_ENTRY_ENVIRONMENT_PRELUDE.to_string());
        }
        if request.is_main && request.format == diffpack_core::ModuleFormat::Esm {
            output
                .entry_preludes
                .push(crate::runtime::SERVER_ENTRY_ENVIRONMENT_PRELUDE.to_string());
        }
        output
    }

    fn flat_entry_prelude(&self, browser_process_shim: bool) -> Option<String> {
        browser_process_shim.then(|| crate::runtime::BROWSER_ENTRY_ENVIRONMENT_PRELUDE.to_string())
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct TanStackSpecialModulePolicy;

impl diffpack_default_loader::module_policy::SpecialModulePolicy for TanStackSpecialModulePolicy {
    fn query_module(
        &self,
        resource: &diffpack_core::ResourceId,
        _target: diffpack_core::transform::Target,
        compile: &mut diffpack_default_loader::module_policy::SyntheticCompiler<'_>,
    ) -> Result<Option<diffpack_default_loader::module::SpecialModule>, String> {
        if diffpack_default_loader::loader::kind(resource)
            == Some(diffpack_default_loader::loader::LoaderKind::TsrSplit)
        {
            return crate::route_split::compile_split_module(resource, |path, source| {
                compile(path, source)
            })
            .map(Some);
        }
        Ok(None)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct TanStackOutputPolicy;

impl diffpack_default_loader::output::OutputIntegrationPolicy for TanStackOutputPolicy {
    fn write_server_runtime(
        &self,
        server_dir: &std::path::Path,
        hmr: bool,
    ) -> Result<Vec<std::path::PathBuf>, String> {
        crate::runtime::write_server_entry(server_dir, hmr)
    }
}
