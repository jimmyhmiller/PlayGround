//! Browser runtime and synthesized-module policies.

use std::path::Path;

/// Browser-side Node environment compatibility shared by every web integration.
pub const BROWSER_PROCESS_PRELUDE: &str = "globalThis.process=globalThis.process||{};globalThis.process.env=globalThis.process.env||{};globalThis.process.env.NODE_ENV=globalThis.process.env.NODE_ENV||\"production\";\n";

#[derive(Debug, Default, Clone, Copy)]
pub struct WebRuntimePolicy;

impl diffpack_core::runtime::RuntimeIntegrationPolicy for WebRuntimePolicy {
    fn configure(
        &self,
        request: diffpack_core::runtime::RuntimePolicyRequest<'_>,
    ) -> Result<diffpack_core::runtime::RuntimePolicyOutput, String> {
        let mut output = diffpack_core::runtime::RuntimePolicyOutput {
            browser_require_native: Some(diffpack_core::runtime::RuntimeContribution::new(
                "browser-require-native",
                "diffpack-web",
                crate::runtime::require_native(),
            )),
            ..Default::default()
        };
        if request.is_main
            && request.format == diffpack_core::ModuleFormat::BrowserEsm
            && request.browser_process_shim
        {
            output
                .entry_preludes
                .push(diffpack_core::runtime::RuntimeContribution::new(
                    "browser-process-compatibility",
                    "diffpack-web",
                    BROWSER_PROCESS_PRELUDE.to_string(),
                ));
        }
        if request.hmr {
            let hot = crate::hmr::registry_render_policy(
                request.entry_runtime_id,
                request.any_async,
                request.format == diffpack_core::ModuleFormat::Esm,
            );
            output.hot = Some(diffpack_core::runtime::RuntimeContribution::new(
                "hmr-runtime",
                "diffpack-web",
                diffpack_core::runtime::OwnedRuntimeHotPolicy {
                    require_dynamic: hot.require_dynamic.to_string(),
                    hot_install: hot.hot_install.to_string(),
                    methods: hot.methods,
                    runtime_return: hot.runtime_return,
                    reimport_guard: hot.reimport_guard.to_string(),
                    server_control: hot.server_control.to_string(),
                },
            ));
        }
        Ok(output)
    }

    fn flat_entry_prelude(&self, browser_process_shim: bool) -> Option<String> {
        browser_process_shim.then(|| BROWSER_PROCESS_PRELUDE.to_string())
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct WebSpecialModulePolicy;

impl diffpack_default_loader::module_policy::SpecialModulePolicy for WebSpecialModulePolicy {
    fn finalize_module(
        &self,
        id: &str,
        target: diffpack_core::transform::Target,
        hmr: bool,
        jsx: diffpack_core::parser::JsxExtensions,
        module: &mut diffpack_default_loader::module::SpecialModule,
    ) {
        if hmr
            && target == diffpack_core::transform::Target::Client
            && crate::hmr::is_refresh_boundary(Path::new(id), &[], "", jsx)
        {
            module.code.push_str(&crate::hmr::fast_refresh_footer(id));
        }
    }
}

#[cfg(test)]
mod tests {
    use diffpack_core::runtime::{RuntimeIntegrationPolicy, RuntimePolicyRequest};

    use super::{BROWSER_PROCESS_PRELUDE, WebRuntimePolicy};

    #[test]
    fn requested_browser_process_compatibility_is_owned_by_the_web_layer() {
        assert_eq!(
            WebRuntimePolicy.flat_entry_prelude(true).as_deref(),
            Some(BROWSER_PROCESS_PRELUDE)
        );
        assert_eq!(WebRuntimePolicy.flat_entry_prelude(false), None);
    }

    #[test]
    fn plain_web_runtime_profile_snapshot() {
        let output = WebRuntimePolicy
            .configure(RuntimePolicyRequest {
                format: diffpack_core::ModuleFormat::BrowserEsm,
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
            .unwrap();
        assert_eq!(
            output.describe(),
            [
                "browser-process-compatibility@diffpack-web",
                "browser-require-native@diffpack-web",
            ]
        );
    }
}
