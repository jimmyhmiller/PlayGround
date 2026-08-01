//! Integration-neutral configuration for the filesystem graph driver.

use std::path::PathBuf;
use std::sync::Arc;

use diffpack_core::compiler::ModuleCompiler;
use diffpack_core::parser::JsxExtensions;
use diffpack_core::runtime::RuntimeIntegrationPolicy;
use diffpack_core::transform::{JsxConfig, Target};

use crate::source_policy::{NoSourceIntegrationPolicy, SourceIntegrationPolicy};
use crate::{CssPreprocess, ImageImportShape};

/// Complete host-owned policy set for the filesystem graph driver.
pub struct DriverPolicies {
    pub compiler: Arc<dyn ModuleCompiler>,
    pub special_modules: Arc<dyn crate::module_policy::SpecialModulePolicy>,
    pub runtime: Arc<dyn RuntimeIntegrationPolicy>,
    pub output: Arc<dyn crate::output::OutputIntegrationPolicy>,
    pub source: Arc<dyn SourceIntegrationPolicy>,
}

#[derive(Debug, Clone)]
pub struct EnvironmentConfig {
    pub environment: String,
    pub build: BuildConfig,
    pub entry: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct BuildConfig {
    pub browser_process_shim: bool,
    pub asset_inline_limit: usize,
    pub base: String,
    pub aliases: Vec<(String, String)>,
    pub conditions: Vec<String>,
    pub main_fields: Vec<String>,
    pub virtual_modules: Vec<(String, String)>,
    pub private_chunk_names: Vec<(String, String)>,
    pub target: Target,
    pub source_policy: Arc<dyn SourceIntegrationPolicy>,
    pub hmr: bool,
    pub source_maps: bool,
    pub scss: crate::sass::ScssOptions,
    pub image_import_shape: ImageImportShape,
    pub css_preprocess: CssPreprocess,
    pub jsx_extensions: JsxExtensions,
    pub jsx: JsxConfig,
    pub server_external_packages: Vec<String>,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            base: "/".to_string(),
            browser_process_shim: false,
            asset_inline_limit: 0,
            aliases: Vec::new(),
            conditions: Vec::new(),
            main_fields: Vec::new(),
            virtual_modules: Vec::new(),
            private_chunk_names: Vec::new(),
            target: Target::default(),
            source_policy: Arc::new(NoSourceIntegrationPolicy),
            hmr: false,
            source_maps: false,
            scss: crate::sass::ScssOptions::default(),
            image_import_shape: ImageImportShape::Url,
            css_preprocess: CssPreprocess::default(),
            jsx_extensions: JsxExtensions::JsxAndTsxOnly,
            jsx: JsxConfig::default(),
            server_external_packages: Vec::new(),
        }
    }
}
