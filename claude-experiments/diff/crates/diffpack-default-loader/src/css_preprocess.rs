//! Configuration for stylesheet preprocessors owned by the default loader.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::postcss::Postcss;

#[derive(Debug, Clone, Default)]
pub struct CssPreprocess {
    pub root: Option<PathBuf>,
    pub postcss: Option<Arc<Postcss>>,
}

impl CssPreprocess {
    pub fn root_path(&self) -> Option<&Path> {
        self.root.as_deref()
    }
}
