//! TanStack-owned route and server-function compilation policy.

use diffpack_core::compiler::{CompileRequest, ModuleCompiler, PreparedSource};
use diffpack_core::graph::BuildUpdate;
use diffpack_core::source_map::MapOrigin;
use diffpack_core::transform::{ModuleLiveness, TransformDiagnostic, TransformResult};
use diffpack_default_loader::driver::Bundler;
use diffpack_default_loader::driver_config::{BuildConfig, DriverPolicies};

#[derive(Debug, Default, Clone, Copy)]
pub struct TanStackCompiler;

impl ModuleCompiler for TanStackCompiler {
    fn compile(&self, request: CompileRequest<'_>) -> TransformResult {
        let mut code = request.source.to_string();
        let mut origin = MapOrigin::File;
        if let Some(rewritten) = crate::route_split::split_reference_route(request.path, &code) {
            code = rewritten;
            origin = MapOrigin::Generated("route-split");
        }
        match crate::server_fn::transform_server_fns(
            request.path,
            &code,
            request.target == diffpack_core::transform::Target::Client,
        ) {
            Ok(Some(rewritten)) => {
                code = rewritten;
                origin = MapOrigin::Generated("server-fn");
            }
            Ok(None) => {}
            Err(error) => return failed(error),
        }
        diffpack_core::compiler::transform_prepared_module_in_language_with(
            request.path,
            PreparedSource {
                code: &code,
                force_jsx: false,
                map_origin: origin,
            },
            request.target,
            request.refresh,
            request.jsx,
            request.project_config,
            request.language,
            request.source_maps,
            &crate::env_transform::TanStackSemanticTransform,
        )
    }
}

fn failed(error: String) -> TransformResult {
    TransformResult {
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
    }
}

/// Discover a TanStack graph with its compiler and runtime policies.
pub fn discover(
    entry: &std::path::Path,
    config: &BuildConfig,
) -> Result<(Bundler, BuildUpdate), String> {
    Bundler::discover_with_driver_policies(
        entry,
        config,
        diffpack_core::ProviderPipeline::default(),
        DriverPolicies {
            compiler: std::sync::Arc::new(diffpack_web::compiler::WebCompiler::new(
                std::sync::Arc::new(TanStackCompiler),
            )),
            special_modules: std::sync::Arc::new(
                diffpack_default_loader::module_policy::SpecialModulePolicyChain::new(vec![
                    std::sync::Arc::new(crate::policies::TanStackSpecialModulePolicy),
                    std::sync::Arc::new(diffpack_web::policies::WebSpecialModulePolicy),
                ]),
            ),
            runtime: std::sync::Arc::new(diffpack_core::runtime::RuntimePolicyChain::new(vec![
                std::sync::Arc::new(crate::policies::TanStackRuntimePolicy),
                std::sync::Arc::new(diffpack_web::policies::WebRuntimePolicy),
            ])),
            output: std::sync::Arc::new(crate::policies::TanStackOutputPolicy),
            source: std::sync::Arc::clone(&config.source_policy),
        },
    )
}
