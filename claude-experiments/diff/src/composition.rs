//! Root policy composition for the integration-neutral filesystem driver.

use std::path::Path;
use std::sync::Arc;

use diffpack_core::ProviderPipeline;
use diffpack_core::compiler::ModuleCompiler;
use diffpack_core::graph::BuildUpdate;
use diffpack_core::runtime::RuntimeIntegrationPolicy;
use diffpack_default_loader::driver_config::{BuildConfig, DriverPolicies};
use diffpack_default_loader::module_policy::SpecialModulePolicy;
use diffpack_default_loader::output::OutputIntegrationPolicy;

use crate::bundler::Bundler;

fn web_policies(config: &BuildConfig) -> DriverPolicies {
    policies(
        config,
        Arc::new(diffpack_web::compiler::WebCompiler::core()),
        Arc::new(diffpack_web::policies::WebSpecialModulePolicy),
        Arc::new(diffpack_web::policies::WebRuntimePolicy),
        Arc::new(diffpack_default_loader::output::NoOutputIntegrationPolicy),
    )
}

pub fn discover_web_with_config(
    entry: &Path,
    config: &BuildConfig,
) -> Result<(Bundler, BuildUpdate), String> {
    diffpack_web::compiler::discover(entry, config)
}

pub fn discover_next_with_config(
    entry: &Path,
    config: &BuildConfig,
) -> Result<(Bundler, BuildUpdate), String> {
    diffpack_next::compiler::discover(entry, config)
}

pub fn discover_tanstack_with_config(
    entry: &Path,
    config: &BuildConfig,
) -> Result<(Bundler, BuildUpdate), String> {
    diffpack_tanstack::compiler::discover(entry, config)
}

fn policies(
    config: &BuildConfig,
    compiler: Arc<dyn ModuleCompiler>,
    special_modules: Arc<dyn SpecialModulePolicy>,
    runtime: Arc<dyn RuntimeIntegrationPolicy>,
    output: Arc<dyn OutputIntegrationPolicy>,
) -> DriverPolicies {
    DriverPolicies {
        compiler,
        special_modules,
        runtime,
        output,
        source: Arc::clone(&config.source_policy),
    }
}

pub fn discover(entry: &Path) -> Result<(Bundler, BuildUpdate), String> {
    discover_direct(entry)
}

pub fn discover_direct(entry: &Path) -> Result<(Bundler, BuildUpdate), String> {
    discover_direct_with_config(entry, &BuildConfig::default())
}

pub fn discover_direct_with_config(
    entry: &Path,
    config: &BuildConfig,
) -> Result<(Bundler, BuildUpdate), String> {
    discover_direct_with_config_and_providers(entry, config, ProviderPipeline::default())
}

pub fn discover_direct_with_config_and_providers(
    entry: &Path,
    config: &BuildConfig,
    providers: ProviderPipeline,
) -> Result<(Bundler, BuildUpdate), String> {
    let selected = web_policies(config);
    Bundler::discover_with_driver_policies(entry, config, providers, selected)
}

pub fn discover_direct_with_config_providers_and_compiler(
    entry: &Path,
    config: &BuildConfig,
    providers: ProviderPipeline,
    compiler: Arc<dyn ModuleCompiler>,
) -> Result<(Bundler, BuildUpdate), String> {
    Bundler::discover_with_driver_policies(
        entry,
        config,
        providers,
        policies(
            config,
            compiler,
            Arc::new(diffpack_web::policies::WebSpecialModulePolicy),
            Arc::new(diffpack_web::policies::WebRuntimePolicy),
            Arc::new(diffpack_default_loader::output::NoOutputIntegrationPolicy),
        ),
    )
}

pub fn discover_with_policies(
    entry: &Path,
    config: &BuildConfig,
    providers: ProviderPipeline,
    compiler: Arc<dyn ModuleCompiler>,
    special_modules: Arc<dyn SpecialModulePolicy>,
) -> Result<(Bundler, BuildUpdate), String> {
    Bundler::discover_with_driver_policies(
        entry,
        config,
        providers,
        policies(
            config,
            compiler,
            special_modules,
            Arc::new(diffpack_web::policies::WebRuntimePolicy),
            Arc::new(diffpack_default_loader::output::NoOutputIntegrationPolicy),
        ),
    )
}

pub fn discover_with_all_policies(
    entry: &Path,
    config: &BuildConfig,
    providers: ProviderPipeline,
    compiler: Arc<dyn ModuleCompiler>,
    special_modules: Arc<dyn SpecialModulePolicy>,
    runtime: Arc<dyn RuntimeIntegrationPolicy>,
) -> Result<(Bundler, BuildUpdate), String> {
    Bundler::discover_with_driver_policies(
        entry,
        config,
        providers,
        policies(
            config,
            compiler,
            special_modules,
            runtime,
            Arc::new(diffpack_default_loader::output::NoOutputIntegrationPolicy),
        ),
    )
}
