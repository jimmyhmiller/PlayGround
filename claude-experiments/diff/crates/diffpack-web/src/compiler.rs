//! Browser development compilation layered over an arbitrary source compiler.

use std::path::Path;
use std::sync::Arc;

use diffpack_core::compiler::{CompileRequest, ModuleCompiler};
use diffpack_core::graph::BuildUpdate;
use diffpack_core::transform::TransformResult;
use diffpack_default_loader::driver::Bundler;
use diffpack_default_loader::driver_config::{BuildConfig, DriverPolicies};

/// Adds `import.meta.hot` and React Fast Refresh instrumentation after the
/// integration-specific compiler has produced ordinary JavaScript.
pub struct WebCompiler {
    inner: Arc<dyn ModuleCompiler>,
}

impl WebCompiler {
    pub fn new(inner: Arc<dyn ModuleCompiler>) -> Self {
        Self { inner }
    }

    pub fn core() -> Self {
        Self::new(Arc::new(diffpack_core::compiler::CoreModuleCompiler))
    }
}

impl ModuleCompiler for WebCompiler {
    fn compile(&self, request: CompileRequest<'_>) -> TransformResult {
        let hmr = request.hmr;
        let refresh = request.refresh;
        let path = request.path;
        let source = request.source;
        let jsx = request.jsx;
        let mut transformed = self.inner.compile(request);
        if hmr {
            let before_refresh = std::mem::take(&mut transformed.code);
            let hot_rewritten = before_refresh.contains("import.meta.hot");
            transformed.code = crate::hmr::rewrite_import_meta_hot(&before_refresh);
            let mut preamble_lines = 0_u32;
            if refresh {
                let module_key = path.to_string_lossy();
                if crate::hmr::needs_fast_refresh_preamble(&transformed.code) {
                    let preamble = crate::hmr::fast_refresh_preamble(&module_key);
                    preamble_lines = preamble.bytes().filter(|byte| *byte == b'\n').count() as u32;
                    transformed.code.insert_str(0, &preamble);
                }
                if crate::hmr::is_refresh_boundary(path, &transformed.liveness.exports, source, jsx)
                {
                    transformed
                        .code
                        .push_str(&crate::hmr::fast_refresh_footer(&module_key));
                }
            }
            crate::hmr::rebase_refresh_map(
                &mut transformed.map,
                &before_refresh,
                preamble_lines,
                hot_rewritten,
            );
        }
        transformed
    }

    fn is_generated_path(&self, path: &Path) -> bool {
        self.inner.is_generated_path(path)
    }

    fn unresolved_import_help(&self, specifier: &str) -> Option<&'static str> {
        self.inner.unresolved_import_help(specifier)
    }
}

/// Discover a framework-neutral browser graph with the web compiler/runtime
/// policies and the default filesystem providers.
pub fn discover(entry: &Path, config: &BuildConfig) -> Result<(Bundler, BuildUpdate), String> {
    Bundler::discover_with_driver_policies(
        entry,
        config,
        diffpack_core::ProviderPipeline::default(),
        DriverPolicies {
            compiler: Arc::new(WebCompiler::core()),
            special_modules: Arc::new(crate::policies::WebSpecialModulePolicy),
            runtime: Arc::new(crate::policies::WebRuntimePolicy),
            output: Arc::new(diffpack_default_loader::output::NoOutputIntegrationPolicy),
            source: Arc::clone(&config.source_policy),
        },
    )
}
