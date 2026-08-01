//! Compatibility facade for integration-owned build configuration.

use std::collections::BTreeSet;
use std::path::Path;

use crate::bundler::{Bundler, ModuleId};

pub use diffpack_default_loader::driver_config::EnvironmentConfig as AppConfig;
pub use diffpack_tanstack::config::ENVIRONMENTS;
pub use diffpack_web::config::{
    WebConfig, copy_static_public, derive_web_config, set_web_development_mode, vite_config_string,
};

pub fn configure_next_app(root: &Path, environment: &str) -> Result<Option<AppConfig>, String> {
    diffpack_next::next_adapter::configure_app_router(root, environment)
}

pub fn configure_next_app_dev(
    root: &Path,
    environment: &str,
    scope: &diffpack_next::next_adapter::RouteScope,
) -> Result<Option<AppConfig>, String> {
    diffpack_next::next_adapter::configure_app_router_dev(root, environment, scope)
}

pub fn configure_next_pages(
    root: &Path,
    environment: &str,
    dev: bool,
) -> Result<Option<AppConfig>, String> {
    diffpack_next::next_pages::configure(root, environment, dev)
}

pub fn reconcile_next_async_islands(
    root: &Path,
    environment: &str,
    bundler: &Bundler,
    reachable: &BTreeSet<ModuleId>,
) -> Result<bool, String> {
    diffpack_next::next_adapter::reconcile_async_islands_from_tainted(
        root,
        environment,
        &bundler.async_tainted_modules(reachable),
    )
}

pub fn derive_config(root: &Path, environment: &str) -> Result<AppConfig, String> {
    diffpack_tanstack::config::derive_config(root, environment)
}

pub fn set_development_mode(config: &mut AppConfig) {
    config.build.hmr = true;
    if let Some(policy) = config.build.source_policy.development() {
        config.build.source_policy = policy;
    }
}
