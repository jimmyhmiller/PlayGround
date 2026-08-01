//! Next-owned source preparation layered over the neutral core compiler.

use std::path::Path;

use diffpack_core::compiler::{CompileRequest, ModuleCompiler, PreparedSource};
use diffpack_core::graph::BuildUpdate;
use diffpack_core::source_map::MapOrigin;
use diffpack_core::transform::{ModuleLiveness, Target, TransformDiagnostic, TransformResult};
use diffpack_default_loader::driver::Bundler;
use diffpack_default_loader::driver_config::{BuildConfig, DriverPolicies};

#[derive(Debug, Default, Clone, Copy)]
pub struct NextCompiler {
    async_client_module_container: bool,
}

impl NextCompiler {
    fn native_next() -> Self {
        Self {
            async_client_module_container: true,
        }
    }
}

impl ModuleCompiler for NextCompiler {
    fn compile(&self, request: CompileRequest<'_>) -> TransformResult {
        let prepared = match prepare(
            request.path,
            request.source,
            request.target,
            self.async_client_module_container,
        ) {
            Ok(prepared) => prepared,
            Err(error) => return failed(error),
        };
        diffpack_core::compiler::transform_prepared_module_in_language(
            request.path,
            PreparedSource {
                code: &prepared.code,
                force_jsx: prepared.force_jsx,
                map_origin: prepared.origin,
            },
            request.target,
            request.refresh,
            request.jsx,
            request.project_config,
            request.language,
            request.source_maps,
        )
    }

    fn is_generated_path(&self, path: &Path) -> bool {
        crate::is_generated_adapter_path(path)
    }

    fn unresolved_import_help(&self, specifier: &str) -> Option<&'static str> {
        (diffpack_default_loader::resolver_policy::bare_package_name(specifier).as_deref()
            == Some(crate::rsc_runtime_resolve::PACKAGE))
        .then_some(crate::rsc_runtime_resolve::MISSING_RUNTIME_HELP)
    }
}

struct Prepared {
    code: String,
    force_jsx: bool,
    origin: MapOrigin,
}

fn prepare(
    path: &Path,
    source: &str,
    target: Target,
    async_client_module_container: bool,
) -> Result<Prepared, String> {
    let mut code = source.to_string();
    let mut force_jsx = false;
    let mut origin = MapOrigin::File;

    if crate::mdx::is_mdx_path(path) {
        code = crate::mdx::compile(path, &code)?.jsx;
        force_jsx = true;
        origin = MapOrigin::Generated("mdx");
    }

    let mut rsc_overridden = false;
    if target == Target::IsolatedServer {
        let rewritten = match crate::rsc::detect_directive(path, &code) {
            Some(crate::rsc::RscDirective::Client) => {
                let mode = if async_client_module_container {
                    crate::rsc::ClientReferenceMode::AsyncModuleContainer
                } else {
                    crate::rsc::ClientReferenceMode::Synchronous
                };
                crate::rsc::transform_use_client_server_with_mode(path, &code, mode)?
            }
            Some(crate::rsc::RscDirective::Server) => {
                crate::rsc::transform_use_server_server(path, &code)?
            }
            Some(crate::rsc::RscDirective::Cache) => {
                crate::rsc::transform_use_cache_server(path, &code)?
            }
            None => None,
        };
        if let Some(rewritten) = rewritten {
            code = rewritten;
            rsc_overridden = true;
            origin = MapOrigin::Generated("rsc-directive");
        }
    }

    if let Some(rewritten) = crate::next_font::transform_next_font(path, &code)? {
        code = rewritten;
        origin = MapOrigin::Generated("next-font");
    }
    if let Some(rewritten) = crate::styled_jsx::transform_styled_jsx(path, &code)? {
        code = rewritten;
    }
    if !rsc_overridden
        && code.contains("use server")
        && crate::rsc::detect_directive(path, &code) == Some(crate::rsc::RscDirective::Server)
    {
        if let Some(rewritten) = match target {
            Target::Client | Target::Server => crate::rsc::transform_use_server_client(path, &code),
            Target::IsolatedServer => crate::rsc::transform_use_server_server(path, &code)?,
        } {
            code = rewritten;
            origin = MapOrigin::Generated("use-server");
        }
    }

    Ok(Prepared {
        code,
        force_jsx,
        origin,
    })
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

/// Discover a Next graph with Next source/module/runtime policy plus the shared
/// browser development layer.
pub fn discover(entry: &Path, config: &BuildConfig) -> Result<(Bundler, BuildUpdate), String> {
    discover_with_compiler(entry, config, NextCompiler::default())
}

/// Discover a graph whose client-reference modules live in the native Next
/// shared SSR container. This changes only Flight's module metadata; the graph,
/// loader, and Next request lifecycle are otherwise identical.
pub fn discover_native_next(
    entry: &Path,
    config: &BuildConfig,
) -> Result<(Bundler, BuildUpdate), String> {
    discover_with_compiler(entry, config, NextCompiler::native_next())
}

fn discover_with_compiler(
    entry: &Path,
    config: &BuildConfig,
    compiler: NextCompiler,
) -> Result<(Bundler, BuildUpdate), String> {
    Bundler::discover_with_driver_policies(
        entry,
        config,
        diffpack_core::ProviderPipeline::default(),
        DriverPolicies {
            compiler: std::sync::Arc::new(diffpack_web::compiler::WebCompiler::new(
                std::sync::Arc::new(compiler),
            )),
            special_modules: std::sync::Arc::new(
                diffpack_default_loader::module_policy::SpecialModulePolicyChain::new(vec![
                    std::sync::Arc::new(crate::policies::NextSpecialModulePolicy),
                    std::sync::Arc::new(diffpack_web::policies::WebSpecialModulePolicy),
                ]),
            ),
            runtime: std::sync::Arc::new(diffpack_core::runtime::RuntimePolicyChain::new(vec![
                std::sync::Arc::new(crate::policies::NextRuntimePolicy),
                std::sync::Arc::new(diffpack_web::policies::WebRuntimePolicy),
            ])),
            output: std::sync::Arc::new(diffpack_default_loader::output::NoOutputIntegrationPolicy),
            source: std::sync::Arc::clone(&config.source_policy),
        },
    )
}
