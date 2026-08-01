//! TanStack-owned runtime, special-module, and output policies.

#[derive(Debug, Default, Clone, Copy)]
pub struct TanStackRuntimePolicy;

impl diffpack_core::runtime::RuntimeIntegrationPolicy for TanStackRuntimePolicy {
    fn configure(
        &self,
        request: diffpack_core::runtime::RuntimePolicyRequest<'_>,
    ) -> Result<diffpack_core::runtime::RuntimePolicyOutput, String> {
        let mut output = diffpack_core::runtime::RuntimePolicyOutput::default();
        if request.is_main
            && request.format == diffpack_core::ModuleFormat::BrowserEsm
            && request.browser_process_shim
        {
            output
                .entry_preludes
                .push(diffpack_core::runtime::RuntimeContribution::new(
                    "tanstack-browser-environment",
                    "diffpack-tanstack",
                    crate::runtime::BROWSER_ENTRY_ENVIRONMENT_PRELUDE.to_string(),
                ));
        }
        if request.is_main && request.format == diffpack_core::ModuleFormat::Esm {
            output
                .entry_preludes
                .push(diffpack_core::runtime::RuntimeContribution::new(
                    "tanstack-server-environment",
                    "diffpack-tanstack",
                    crate::runtime::SERVER_ENTRY_ENVIRONMENT_PRELUDE.to_string(),
                ));
        }
        Ok(output)
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

#[cfg(test)]
mod tests {
    use diffpack_core::runtime::{
        RuntimeIntegrationPolicy, RuntimePolicyChain, RuntimePolicyRequest,
    };

    use super::TanStackRuntimePolicy;

    fn snapshot(format: diffpack_core::ModuleFormat) -> Vec<String> {
        RuntimePolicyChain::new(vec![
            std::sync::Arc::new(TanStackRuntimePolicy),
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
            modules: &[],
            browser_process_shim: true,
        })
        .unwrap()
        .describe()
    }

    #[test]
    fn tanstack_client_and_server_runtime_profile_snapshots() {
        assert_eq!(
            snapshot(diffpack_core::ModuleFormat::BrowserEsm),
            [
                "tanstack-browser-environment@diffpack-tanstack",
                "browser-process-compatibility@diffpack-web",
                "browser-require-native@diffpack-web",
            ]
        );
        assert_eq!(
            snapshot(diffpack_core::ModuleFormat::Esm),
            [
                "tanstack-server-environment@diffpack-tanstack",
                "browser-require-native@diffpack-web",
            ]
        );
    }
}
