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
pub mod browser_field;
pub mod css;
pub mod css_preprocess;
pub mod driver;
pub mod driver_config;
pub mod dynamic_import_context;
pub mod font_file;
pub mod jsx_project_config;
pub mod less_stylus;
pub mod loader;
pub mod module;
pub mod module_policy;
pub mod output;
pub mod postcss;
pub mod resolution_diagnostic;
pub mod resolver;
pub mod resolver_policy;
pub mod runtime;
pub mod runtime_helpers;
pub mod sass;
pub mod sfc;
pub mod side_effects;
pub mod source_policy;
pub mod tailwind;
pub mod tailwind_delegate;
pub mod tailwind_project;

pub use asset::ImageImportShape;
pub use css_preprocess::CssPreprocess;

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
        }))
    }
}
