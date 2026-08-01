//! Public contracts for chunk rendering and emission.

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ModuleFormat {
    #[default]
    Cjs,
    Esm,
    BrowserEsm,
}

impl ModuleFormat {
    pub fn is_esm(self) -> bool {
        matches!(self, Self::Esm | Self::BrowserEsm)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EmitOptions {
    pub source_map: bool,
    pub minify: bool,
    pub format: ModuleFormat,
    pub hmr: bool,
}
