//! Default module-provider layer.
//!
//! The existing native loaders are being migrated here incrementally. This first
//! slice supplies the filesystem source boundary without pulling framework policy
//! into `diffpack-core`.

extern crate self as diffpack_default_loader;

use std::fs;
use std::path::{Path, PathBuf};

use diffpack_core::{
    LoadRequest, LoadedSource, ModuleProvider, ProviderDiagnostic, ResourceId, SourceLanguage,
};

pub mod asset;
#[cfg_attr(feature = "legacy-driver-tests", doc(hidden))]
#[cfg_attr(feature = "legacy-driver-tests", allow(missing_docs))]
#[cfg(feature = "legacy-driver-tests")]
pub mod browser_field;
#[cfg(not(feature = "legacy-driver-tests"))]
mod browser_field;
mod css;
mod css_preprocess;
pub mod define;
pub mod driver;
pub mod driver_config;
#[cfg(feature = "legacy-driver-tests")]
#[doc(hidden)]
pub mod dynamic_import_context;
#[cfg(not(feature = "legacy-driver-tests"))]
mod dynamic_import_context;
mod engine;
pub mod env_file;
pub mod font_file;
#[cfg(feature = "legacy-driver-tests")]
#[doc(hidden)]
pub mod jsx_project_config;
#[cfg(not(feature = "legacy-driver-tests"))]
mod jsx_project_config;
mod less_stylus;
pub mod loader;
pub mod module;
pub mod module_policy;
pub mod output;
pub mod postcss;
#[cfg(feature = "legacy-driver-tests")]
#[doc(hidden)]
pub mod resolution_diagnostic;
#[cfg(not(feature = "legacy-driver-tests"))]
mod resolution_diagnostic;
#[cfg(feature = "legacy-driver-tests")]
#[doc(hidden)]
pub mod resolver;
#[cfg(not(feature = "legacy-driver-tests"))]
mod resolver;
pub mod resolver_policy;
#[cfg(feature = "legacy-driver-tests")]
#[doc(hidden)]
pub mod runtime;
#[cfg(not(feature = "legacy-driver-tests"))]
mod runtime;
#[cfg(feature = "legacy-driver-tests")]
#[doc(hidden)]
pub mod runtime_helpers;
#[cfg(not(feature = "legacy-driver-tests"))]
mod runtime_helpers;
pub mod sass;
mod sfc;
#[cfg(feature = "legacy-driver-tests")]
#[doc(hidden)]
pub mod side_effects;
#[cfg(not(feature = "legacy-driver-tests"))]
mod side_effects;
pub mod source_policy;
pub mod tailwind;
#[cfg(feature = "legacy-driver-tests")]
#[doc(hidden)]
pub mod tailwind_delegate;
#[cfg(not(feature = "legacy-driver-tests"))]
mod tailwind_delegate;
#[cfg(feature = "legacy-driver-tests")]
#[doc(hidden)]
pub mod tailwind_project;
#[cfg(not(feature = "legacy-driver-tests"))]
mod tailwind_project;

pub use asset::ImageImportShape;
pub use css_preprocess::CssPreprocess;
pub use engine::{BuildEngine, BuildEngineBuilder};

#[derive(Debug, Clone)]
pub struct FilesystemProvider {
    root: PathBuf,
}

impl FilesystemProvider {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn language(path: &Path) -> Option<SourceLanguage> {
        match path.extension().and_then(|value| value.to_str()) {
            Some("js" | "jsx" | "mjs" | "cjs") => Some(SourceLanguage::JavaScript),
            Some("ts" | "tsx" | "mts" | "cts") => Some(SourceLanguage::TypeScript),
            Some("json") => Some(SourceLanguage::Json),
            Some("css") => Some(SourceLanguage::Text),
            _ => None,
        }
    }
}

impl ModuleProvider for FilesystemProvider {
    fn name(&self) -> &str {
        "diffpack:filesystem"
    }

    fn load(&self, request: LoadRequest<'_>) -> Result<Option<LoadedSource>, ProviderDiagnostic> {
        let resource = ResourceId::parse(request.id);
        let path = Path::new(&resource.path);
        let path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.root.join(path)
        };
        let Some(language) = Self::language(&path) else {
            return Ok(None);
        };
        let code = fs::read(&path).map_err(|error| ProviderDiagnostic {
            message: format!("cannot read {}: {error}", path.display()),
            provider: Some(self.name().to_string()),
        })?;
        Ok(Some(LoadedSource {
            code,
            language,
            source_map: None,
            watch_files: vec![path],
            diagnostics: Vec::new(),
        }))
    }
}
